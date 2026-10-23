"""The offline cascade baseline, on the same Mac and clips as run.py: the recognizer transcribes the
finished recording, then the same LLM reads the transcript and answers.

  python bench/cascade.py clips/ --asr mlx-community/parakeet-tdt_ctc-110m --llm LLM --out parakeet.jsonl
  python bench/cascade.py clips/ --asr mlx-community/whisper-tiny --llm LLM --out tiny.jsonl
  python bench/cascade.py clips/ --text refs.jsonl --llm LLM --out text.jsonl     # no speech: upper bound

An --asr name with "parakeet" is a parakeet-mlx model, any other an mlx-whisper model (mlx-whisper pulls in PyTorch:
use a separate environment). Latency runs from the end of the recording to the first answer token. --text REFS reads
each file's prompt from REFS ("file", "prompt") instead of the audio; --transcribe stops at the transcript.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import audio, llm  # noqa: E402
from listen.stream import DEFAULT_LLM, SAMPLE_RATE, SYSTEM  # noqa: E402


def recognizer(name: str):
    """-> a function from 16 kHz audio to text."""
    if "parakeet" in name:
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        pk = from_pretrained(name)
        return lambda x: pk.generate(get_logmel(mx.array(x), pk.preprocessor_config))[0].text
    import mlx_whisper

    return lambda x: mlx_whisper.transcribe(x, path_or_hf_repo=name, language="en")["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--asr", default="mlx-community/parakeet-tdt_ctc-110m")
    ap.add_argument("--llm", default=DEFAULT_LLM, help="any mlx-lm chat model (Hugging Face id or directory)")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--transcribe", action="store_true", help="report the transcript only (no LLM)")
    ap.add_argument("--text", metavar="REFS", help="read the written prompt from this refs.jsonl instead of any audio")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = (None, None) if args.transcribe else llm.load(args.llm)
    prompts = {r["file"]: r["prompt"] for r in map(json.loads, open(args.text)) if "prompt" in r} if args.text else {}
    asr = None if args.text else recognizer(args.asr)
    files = audio.expand(args.inputs)
    if asr is not None:
        asr(audio.load(files[0], SAMPLE_RATE))  # load + compile outside the timing
    with open(args.out, "w") as out:
        for path in files:
            x = audio.load(path, SAMPLE_RATE)
            t_end = time.perf_counter()
            transcript = prompts[path.name] if asr is None else asr(x).strip()
            t_asr = time.perf_counter()
            text, first = transcript, t_asr
            if model is not None:
                ids = tok.encode(llm.chat_text(tok, args.system, transcript), add_special_tokens=False)
                cache = llm.new_cache(model)
                logits = llm.forward(model, llm.embed(model, ids), cache, last_only=True)
                pieces = []
                for piece in llm.greedy(model, tok, logits, cache, args.max_tokens):
                    if not pieces:
                        first = time.perf_counter()
                    pieces.append(piece)
                text = "".join(pieces).strip()
            t_done = time.perf_counter()
            rec = {"file": path.name, "seconds": len(x) / SAMPLE_RATE, "text": text, "transcript": transcript,
                   "asr_ms": (t_asr - t_end) * 1e3, "first_token_ms": (first - t_end) * 1e3 if model else None,
                   "answer_s": t_done - t_end, "mode": "transcribe" if args.transcribe else "text" if args.text
                   else "cascade", "asr": None if args.text else args.asr, "llm": args.llm}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if model is None:
                print(f"{path.name}: ASR latency {rec['asr_ms']:.0f} ms (no LLM) | {text[:80]}")
            else:
                print(f"{path.name}: asr {rec['asr_ms']:.0f} ms, first token {rec['first_token_ms']:.0f} ms "
                      f"| {text[:80]}")
    mx.clear_cache()


if __name__ == "__main__":
    main()
