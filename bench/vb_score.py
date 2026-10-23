"""BBH and IFEval scored by VoiceBench's own evaluators. Needs a VoiceBench clone (github.com/MatthewCYM/VoiceBench)
and its requirements, in a separate environment:

  VOICEBENCH=path/to/VoiceBench python bench/vb_score.py bbh REFS cascade.jsonl stream.jsonl ... [--tasks]

Each file gets its official score and, after the first, the paired difference with a bootstrap 95 % interval.
VoiceBench's BBH scorer guesses at random when it finds no answer: guesses are seeded and counted. --tasks adds, per
BBH task, accuracy, unparsed answers and accuracy among parsed answers.
"""
import json
import os
import random
import sys

import numpy as np
from langdetect import DetectorFactory

sys.path.insert(0, os.environ.get("VOICEBENCH", "VoiceBench"))
sys.modules["qa_metrics.pedant"] = type(sys)("qa_metrics.pedant")  # only VoiceBench's open-QA scorer needs it
sys.modules["qa_metrics.pedant"].PEDANT = None
from src.evaluator.bbh import BBHEvaluator  # noqa: E402
from src.evaluator.ifeval import IFEvaluator, read_prompt_list, test_instruction_following_strict  # noqa: E402

DetectorFactory.seed = 0  # IFEval's language check is random unless seeded
GUESSES = [0]
_choice = random.choice
random.choice = lambda xs: GUESSES.__setitem__(0, GUESSES[0] + 1) or _choice(xs)


def correct(task, items) -> tuple[np.ndarray, np.ndarray]:
    """Per item: right (BBH answer, or every IFEval instruction followed, strict) and parsed (no random guess)."""
    if task == "bbh":
        ev, gt, right, parsed = BBHEvaluator(), {"yes": 1, "no": 0, "(a)": 0, "(b)": 1}, [], []
        for i in items:
            g = GUESSES[0]
            right.append(ev.extract_answer(i["response"], i["id"]) == gt[i["reference"].lower()])
            parsed.append(GUESSES[0] == g)
        return np.array(right, float), np.array(parsed)
    resp = {i["prompt"]: i["response"] for i in items}
    right = [test_instruction_following_strict(x, resp).follow_all_instructions for x in read_prompt_list(items)]
    return np.array(right, float), np.ones(len(right), bool)


def main(task, refs_path, *paths):
    tasks, paths = "--tasks" in paths, [p for p in paths if p != "--tasks"]
    refs = {r["file"]: r for r in map(json.loads, open(refs_path))}
    runs = [{r["file"]: r["text"] for r in map(json.loads, open(p))} for p in paths]
    files = sorted(set(refs).intersection(*runs))
    first = None
    for p, run in zip(paths, runs, strict=True):
        items = [{**refs[f], "response": run[f]} for f in files]
        random.seed(0)
        official = (BBHEvaluator() if task == "bbh" else IFEvaluator()).evaluate(items)
        GUESSES[0] = 0
        random.seed(0)  # the same guesses again, so per-item results match the official score
        right, parsed = correct(task, items)
        line = f"{p}: {len(files)} items | official {json.dumps({k: round(v, 3) for k, v in official.items()})}"
        line += f" | no answer found (random guess) {GUESSES[0]}" if task == "bbh" else ""
        if first is None:
            first = right
        else:
            d = right - first
            idx = np.random.default_rng(0).integers(0, len(d), (10_000, len(d)))
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            line += f" | vs {paths[0]}: {100 * d.mean():+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})"
        print(line)
        for t in sorted({f.rsplit("_", 1)[0] for f in files}) if tasks else ():
            k = np.array([f.rsplit("_", 1)[0] == t for f in files])
            print(f"  {t}: accuracy {100 * right[k].mean():.1f} %, no answer {int((~parsed[k]).sum())}, "
                  f"accuracy when parsed {100 * right[k & parsed].mean():.1f} %")


if __name__ == "__main__":
    main(*sys.argv[1:])
