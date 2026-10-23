"""Two checks behind numbers the report gives in its text rather than in a table.

  python bench/checks.py qend                                          # q_end, complete vs unfinished question
  python bench/checks.py chunking logs/e5_lcp.jsonl --n 20            # one pass vs 5-token chunks, no rewinds

qend: P(bare "?") and q_end after the chat prefix and the question. chunking: each logged final transcript read in one
pass vs in chunks, never rewinding (first token, greedy answer, logits; bfloat16 and float32).
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import llm  # noqa: E402
from listen.stream import DEFAULT_LLM, SYSTEM, closing_ids  # noqa: E402


def qend(model, tok, questions=("What is the capital of France", "What is the capital of")):
    before, _ = llm.chat_pieces(tok, SYSTEM, "")
    closing, mark = closing_ids(tok), tok.encode("?", add_special_tokens=False)[0]
    for dtype in (mx.bfloat16, mx.float32):
        model.set_dtype(dtype)
        for q in questions:
            cache = llm.new_cache(model)
            llm.prefill(model, llm.embed(model, before), cache)
            ids = tok.encode(q, add_special_tokens=False)
            p = mx.softmax(llm.forward(model, llm.embed(model, ids), cache, last_only=True)[0, -1])
            print(f"{dtype}: {q!r}: bare '?' {p[mark].item():.3f}, q_end {p[closing].sum().item():.3f} "
                  f"({len(closing)} tokens)")


def read(model, tok, text: str, chunk: int | None, max_tokens: int):
    """Chat prefix, the transcript (in one pass or in chunks), the chat suffix; -> first-token logits, answer."""
    before, after = llm.chat_pieces(tok, SYSTEM, "")
    ids = tok.encode(text, add_special_tokens=False)
    cache = llm.new_cache(model)
    llm.prefill(model, llm.embed(model, before), cache)
    for i in range(0, len(ids), chunk or len(ids)):
        llm.prefill(model, llm.embed(model, ids[i : i + (chunk or len(ids))]), cache)
    logits = llm.forward(model, llm.embed(model, after), cache, last_only=True)[0, -1]
    first = int(mx.argmax(logits).item())
    return logits, first, "".join(llm.decode(model, tok, first, cache, max_tokens))


def chunking(model, tok, records: str, n: int, chunk: int, max_tokens: int):
    texts = [json.loads(line)["transcript"] for line in open(records)][:n]
    for dtype in (mx.bfloat16, mx.float32):
        model.set_dtype(dtype)
        top1 = same = 0
        diff = 0.0
        for text in texts:
            a_logits, a_first, a_answer = read(model, tok, text, None, max_tokens)
            b_logits, b_first, b_answer = read(model, tok, text, chunk, max_tokens)
            top1, same = top1 + (a_first == b_first), same + (a_answer == b_answer)
            diff = max(diff, mx.abs(a_logits - b_logits).max().item())
        print(f"{dtype}: one pass vs {chunk}-token chunks, {len(texts)} transcripts: same first token {top1}/"
              f"{len(texts)}, same {max_tokens}-token answer {same}/{len(texts)}, max |dlogit| {diff:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["qend", "chunking"])
    ap.add_argument("records", nargs="?", help="chunking: a run.py records file; its final transcripts are read")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=128)
    args = ap.parse_args()
    model, tok = llm.load(args.llm)
    if args.check == "qend":
        qend(model, tok)
    else:
        chunking(model, tok, args.records, args.n, args.chunk, args.max_tokens)


if __name__ == "__main__":
    main()
