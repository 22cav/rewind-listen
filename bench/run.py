"""The one runner: stream audio files through listen.stream.Listener and write one JSON record per file (transcript,
answer, clocks, the step log and the 20 ms frame log). Every experiment is a run of this script plus a function of
analyze.py.

  python bench/run.py clips/ --out stream.jsonl                       # live-paced, turn ends at the last sample
  python bench/run.py clips/ --fast --parity --out e1.jsonl          # unpaced; check against an offline read
  python bench/run.py clips/ --fast --probe --out e3.jsonl           # only q_end at the end of each clip
  python bench/run.py clips/ --endpoint --out e5.jsonl               # the turn ends by itself (silence + q_end)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import audio, llm  # noqa: E402
from listen.stream import DEFAULT_ASR, DEFAULT_LLM, FRAME, SAMPLE_RATE, SYSTEM, Listener, Parakeet  # noqa: E402


def run(lis: Listener, x, args) -> dict:
    lis.start()
    t_origin, stop = time.perf_counter(), None  # sample i arrives at t_origin + (i + 1) / SAMPLE_RATE
    for i in range(0, len(x), FRAME):
        if not args.fast:
            time.sleep(max(0.0, t_origin + (i + FRAME) / SAMPLE_RATE - time.perf_counter()))
        if lis.feed(x[i : i + FRAME]) and args.endpoint:
            stop = (i + FRAME) / SAMPLE_RATE
            break
    # the turn ends: last sample fed, or the endpointer fired; paced, at that frame's arrival time, even if a pass
    # was still running then (otherwise its delay would hide inside the feed loop)
    t_end = time.perf_counter() if args.fast else t_origin + (stop if stop else len(x) / SAMPLE_RATE)
    rec = {"seconds": len(x) / SAMPLE_RATE, "stop_s": stop}
    if args.probe:  # one pass over all the audio, q_end on the unpunctuated transcript, no answer
        lis.pass_()
        if args.parity:
            rec["offline_transcript"] = " ".join(w[0] for w in lis.asr(x))
        return rec | {"transcript": lis.transcript, **lis.log()}
    clock, pieces = {}, []
    for piece in lis.finish(clock, args.max_tokens):
        pieces.append(piece)
    rec |= {"transcript": lis.transcript, "text": "".join(pieces).strip(),
            "first_token_ms": 1e3 * (clock["token1"] - t_end), "final_ms": 1e3 * (clock["synced"] - t_end),
            "suffix_ms": 1e3 * (clock["token1"] - clock["synced"]), "answer_s": time.perf_counter() - t_end}
    if args.parity:  # exactness: the same transcript read from scratch, offline
        ids = lis.before + lis.tok.encode(lis.transcript, add_special_tokens=False) + lis.after
        cache = llm.new_cache(lis.model)
        ref = llm.forward(lis.model, llm.embed(lis.model, ids), cache, last_only=True)
        tok = int(mx.argmax(ref[0, -1]).item())
        answer = "".join(llm.decode(lis.model, lis.tok, tok, cache, args.max_tokens)).strip()
        rec |= {"parity_max_abs": mx.abs(ref - lis.logits).max().item(),
                "parity_top1": tok == int(mx.argmax(lis.logits[0, -1]).item()),
                "parity_same_answer": answer == rec["text"],
                "offline_transcript": " ".join(w[0] for w in lis.asr(x))}
    return rec | lis.log()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", default=DEFAULT_LLM, help="any mlx-lm chat model")
    ap.add_argument("--asr", default=DEFAULT_ASR, help="a parakeet-mlx model")
    ap.add_argument("--system", default=SYSTEM)
    ap.add_argument("--policy", default="lcp", choices=["lcp", "la2"])
    ap.add_argument("--step", type=float, default=4.0, help="s of speech before a pass with no pause")
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--fast", action="store_true", help="unpaced (accuracy and logs; latencies are not live)")
    ap.add_argument("--endpoint", action="store_true", help="end the turn by the endpointer, not at the last sample")
    ap.add_argument("--parity", action="store_true", help="also read the final transcript offline and compare")
    ap.add_argument("--probe", action="store_true", help="only the step log and q_end at the end of each clip")
    ap.add_argument("--limit", type=int, help="first N files only")
    ap.add_argument("--fp32", action="store_true", help="float32 activations (exact parity; bf16 chunking is not)")
    args = ap.parse_args()
    lis = Listener(args.llm, Parakeet(args.asr), args.system, args.policy, args.step, args.window)
    if args.fp32:
        lis.model.set_dtype(mx.float32)
    files = audio.expand(args.inputs)[: args.limit]
    warm = argparse.Namespace(**{**vars(args), "fast": True, "parity": False})
    run(lis, audio.load(files[0])[: 2 * SAMPLE_RATE], warm)  # load and compile outside any timing
    mx.reset_peak_memory()
    with open(args.out, "w") as out:
        for k, path in enumerate(files):
            rec = {"file": path.name, **run(lis, audio.load(path), args), "llm": args.llm, "asr": args.asr,
                   "policy": args.policy, "step": args.step, "window": args.window, "paced": not args.fast}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{k + 1}/{len(files)}] {path.name}: {rec.get('first_token_ms', 0):.0f} ms | "
                  f"{rec['transcript'][-60:]!r} -> {rec.get('text', '')[:50]!r}", flush=True)
    print(f"peak MLX memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
