"""The report's numbers, as small functions over the records of run.py / cascade.py.

  python bench/analyze.py stability RUN.jsonl ...           exactness and revisions: rewinds, amplification, parity
  python bench/analyze.py fidelity REFS RUN.jsonl ...       WER vs the offline transcript and the reference
  python bench/analyze.py completion CALIB TEST             AUROC of q_end, complete vs incomplete turns
  python bench/analyze.py endpoint LABELS RUN.jsonl         cut-off rate vs end wait, per policy
  python bench/analyze.py latency RUN.jsonl ...             end of turn -> first token
  python bench/analyze.py accuracy REFS A.jsonl B.jsonl ... VoiceBench accuracy, paired vs the first file
  python bench/analyze.py numbers REFS RUN.jsonl ...        accuracy on questions with and without digits
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import mcq  # noqa: E402


def load(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def by_file(path) -> dict:
    return {r["file"]: r for r in load(path)}


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " ")).split()


def edits(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(pairs) -> float:
    """Corpus WER (%) over (reference, hypothesis) pairs."""
    pairs = [(words(r), words(h)) for r, h in pairs]
    return 100 * sum(edits(r, h) for r, h in pairs) / max(1, sum(len(r) for r, _ in pairs))


def ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Paired bootstrap 95 % interval of the mean of per-item differences x."""
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided McNemar p-value for paired right/wrong vectors."""
    from math import comb

    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n, k = n01 + n10, min(n01, n10)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def stability(*paths):
    for p in paths:
        recs = load(p)
        steps = [s for r in recs for s in r["steps"] if not s["final"]]
        amp = [sum(s["adv"] for s in r["steps"]) / max(1, r["steps"][-1]["n"]) for r in recs]
        scratch = [sum(s["n"] for s in r["steps"]) / max(1, r["steps"][-1]["n"]) for r in recs]  # no rewind
        work = np.array([s["asr_ms"] + s["sync_ms"] for s in steps])
        line = (f"{p}: {len(recs)} turns, {len(steps)} steps | rewind median {np.median([s['r'] for s in steps]):.0f} "
                f"p90 {np.percentile([s['r'] for s in steps], 90):.0f} | advance p95 "
                f"{np.percentile([s['adv'] for s in steps], 95):.0f} | amplification median {np.median(amp):.2f} (from "
                f"scratch {np.median(scratch):.2f}) | step work p50 {np.median(work):.0f} p95 "
                f"{np.percentile(work, 95):.0f} ms, total {work.sum() / 1e3:.0f} s")
        if "parity_top1" in recs[0]:
            line += (f" | parity: top-1 {100 * np.mean([r['parity_top1'] for r in recs]):.1f} %, same answer "
                     f"{100 * np.mean([r['parity_same_answer'] for r in recs]):.1f} %, max |dlogit| "
                     f"{max(r['parity_max_abs'] for r in recs):.3f}")
        print(line)


def fidelity(refs_path, *paths):
    refs = by_file(refs_path)
    for p in paths:
        recs = load(p)
        ref = lambda r: refs[r["file"]].get("text") or refs[r["file"]]["prompt"]
        print(f"{p}: WER vs offline {wer((r['offline_transcript'], r['transcript']) for r in recs):.2f} % | vs "
              f"reference: stream {wer((ref(r), r['transcript']) for r in recs):.2f} %, offline "
              f"{wer((ref(r), r['offline_transcript']) for r in recs):.2f} % | cuts per turn "
              f"{np.mean([r['cuts'] for r in recs]):.2f}, forced {sum(r['forced_cuts'] for r in recs)}")


def auroc(pos, neg) -> float:
    """P(score of a positive > score of a negative), ties half."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def completion(calib_dir, test_dir):
    """q_end from the probe pass (punctuation stripped) vs the recognizer's own final punctuation as a detector."""
    scores = {"q_end": lambda r: r["steps"][-1]["q"], "q_end + newline": lambda r: r["steps"][-1]["q_nl"],
              "recognizer punctuation": lambda r: float(bool(re.search(r"[.?!]$", r["transcript"])))}
    for name, d in (("calibration", calib_dir), ("held-out", test_dir)):
        labels, recs = by_file(Path(d) / "labels.jsonl"), load(Path(d) / "probe.jsonl")
        for key, f in scores.items():
            pos = [f(r) for r in recs if labels[r["file"]]["complete"]]
            neg = [f(r) for r in recs if not labels[r["file"]]["complete"]]
            print(f"{name}: {key}: AUROC {auroc(pos, neg):.3f} | median complete {np.median(pos):.3f}, "
                  f"incomplete {np.median(neg):.3f} (n {len(pos)} + {len(neg)})")


def stop_time(rec, tau_of) -> float | None:
    """Replay the endpointer frame by frame on a logged turn: the first frame where trailing silence reaches
    tau_of(q_end), with q_end from the latest step at or before that frame and a non-empty transcript."""
    steps, run, j, q, n = rec["steps"], 0, 0, 0.0, 0
    for i, (db, floor) in enumerate(zip(rec["db"], rec["floor"], strict=True)):
        t = (i + 1) * 0.02
        while j < len(steps) and steps[j]["t"] <= t + 1e-9:
            q, n, j = steps[j]["q"], steps[j]["n"], j + 1
        run = run + 1 if db < floor + 10 else 0
        if n and 20 * run >= tau_of(q):
            return t
    return None


def endpoint(labels_path, *paths):
    """Cut-off rate (stops before the end of speech), end wait (stop - end of speech) and no-stop rate, for fixed and
    q_end-modulated waits; the same logged turns replayed under every policy. Each modulated policy is compared with
    the fixed wait that has the same cut-off rate (linear interpolation on the fixed curve)."""
    labels = by_file(labels_path)
    policies = {f"fixed {t} ms": (lambda q, t=t: t) for t in range(300, 1300, 100)}
    for g in (0.5, 1.0, 2.0):
        for lo in (200, 300, 400):
            policies[f"q_end, gamma {g}, {lo}-1200 ms"] = lambda q, g=g, lo=lo: 1200 - (1200 - lo) * q**g
    for p in paths:
        recs, fixed = load(p), []
        for name, tau in policies.items():
            stops = [(stop_time(r, tau), labels[r["file"]]["speech_end"]) for r in recs]
            cut = 100 * np.mean([s is not None and s < e for s, e in stops])
            wait = [1e3 * (s - e) for s, e in stops if s is not None and s >= e]
            med = np.median(wait) if wait else float("nan")
            line = (f"{p}: {name}: cut-off {cut:.1f} % | end wait median {med:.0f} ms | no stop "
                    f"{100 * np.mean([s is None for s, _ in stops]):.1f} %")
            if name.startswith("fixed"):
                fixed.append((cut, med))
            elif wait and min(c for c, _ in fixed) <= cut <= max(c for c, _ in fixed):
                c, m = zip(*sorted(fixed), strict=True)
                same = float(np.interp(cut, c, m))
                line += f" | fixed wait at the same cut-off {same:.0f} ms: {100 * (1 - med / same):.0f} % shorter"
            print(line)


def latency(*paths):
    for p in paths:
        recs = load(p)
        pct = lambda k, recs=recs: np.percentile([r[k] for r in recs if r.get(k) is not None], [50, 95])
        line = f"{p}: first token p50 {pct('first_token_ms')[0]:.0f} p95 {pct('first_token_ms')[1]:.0f} ms"
        for k in ("final_ms", "suffix_ms", "asr_ms"):
            if recs[0].get(k) is not None:
                line += f" | {k[:-3]} p50 {pct(k)[0]:.0f} ms"
        if "steps" in recs[0]:  # turns where speech came after the last pass, so the recognizer ran once more
            line += f" | final recognizer pass {sum(any(s['final'] for s in r['steps']) for r in recs)}/{len(recs)}"
        if "db" in recs[0]:  # clip length, and the silence the clip ends with (the endpointer's own measure)
            tail = [20 * next((n for n, (d, f) in enumerate(zip(r["db"][::-1], r["floor"][::-1], strict=True))
                               if d >= f + 10), len(r["db"])) for r in recs]
            line += (f" | clip p50 {np.median([r['seconds'] for r in recs]):.1f} s, "
                     f"ends in {np.median(tail):.0f} ms silence")
        print(line)


def numbers(refs_path, *paths):
    """Strict accuracy on the questions whose written prompt contains a digit, and on the others."""
    refs = by_file(refs_path)
    for p in paths:
        s = by_file(p)
        for name, digits in (("with digits", True), ("without digits", False)):
            fs = [f for f in s if f in refs and bool(re.search(r"\d", refs[f]["prompt"])) == digits]
            acc = np.mean([mcq.extract(s[f]["text"]) == refs[f]["reference"] for f in fs])
            print(f"{p}: {name}: {len(fs)} questions, strict {100 * acc:.1f} %")


def accuracy(refs_path, *paths):
    """VoiceBench accuracy (strict and official); every file after the first is compared with the first: paired
    95 % interval of the difference, McNemar p, share of identical answers."""
    refs = by_file(refs_path)
    sets = [by_file(p) for p in paths]
    files = sorted(set(refs) & set.intersection(*(set(s) for s in sets)))
    right = []
    for p, s in zip(paths, sets, strict=True):
        m = mcq.scores([s[f]["text"] for f in files], [refs[f]["reference"] for f in files])
        right.append(np.array([mcq.extract(s[f]["text"]) == refs[f]["reference"] for f in files], float))
        line = f"{p}: {len(files)} questions | strict {100 * m['strict']:.1f} %, official {100 * m['official']:.1f} %"
        if len(right) > 1:
            lo, hi = ci(right[-1] - right[0])
            same = np.mean([mcq.extract(s[f]["text"]) == mcq.extract(sets[0][f]["text"]) for f in files])
            line += (f" | vs {paths[0]}: {100 * (right[-1] - right[0]).mean():+.1f} ({100 * lo:+.1f} to "
                     f"{100 * hi:+.1f}), McNemar p {mcnemar(right[0], right[-1]):.2f}, same choice {100 * same:.1f} %")
        print(line)


if __name__ == "__main__":
    {"stability": stability, "fidelity": fidelity, "completion": completion, "endpoint": endpoint,
     "latency": latency, "accuracy": accuracy, "numbers": numbers}[sys.argv[1]](*sys.argv[2:])
