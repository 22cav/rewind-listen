"""Build the evaluation sets as folders of 16 kHz wav files + one JSONL of labels, from public datasets.

  python bench/data.py voicebench OUT     # VoiceBench multiple choice, dev split: OUT/qa/*.wav + OUT/refs.jsonl
  python bench/data.py pack OUT [LTTS]   # latency: the first 40 OpenBookQA questions (OUT/qa); WER: the 50 LibriTTS-R
                                          # test-clean clips of bench/librittsr50.jsonl, found under LTTS (OUT/asr)
  python bench/data.py smartturn OUT      # Smart Turn v3.2 test, English, real voices: calib/ (200) + test/ (500)
  python bench/data.py pauses OUT         # open-ended questions with one inserted pause, + a multiple-choice stratum
                                          # (needs OUT/../vb_dev from `voicebench`)
  python bench/data.py bbh OUT            # VoiceBench BBH (1,000) or IFEval (345, with `ifeval`), all items, for
                                          # VoiceBench's own scorers: OUT/qa/*.wav + OUT/refs.jsonl

Sources: hlt-lab/voicebench (Apache-2.0), pipecat-ai/smart-turn-data-v3.2-test (CC-BY-4.0), LibriTTS-R (CC-BY-4.0).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import audio  # noqa: E402
from listen.splits import split_of  # noqa: E402
from listen.stream import FRAME, SAMPLE_RATE, Parakeet, Silence  # noqa: E402

VB = "hlt-lab/voicebench"
MMSU = ["biology", "business", "chemistry", "economics", "engineering", "health", "history", "law", "other",
        "philosophy", "physics", "psychology"]
MCQ_FILES = ["openbookqa/test-00000-of-00001.parquet"] + [f"mmsu/{d}-00000-of-00001.parquet" for d in MMSU]


def rows(repo: str, file: str):
    return pq.read_table(hf_hub_download(repo, file, repo_type="dataset")).to_pylist()


def wav16(blob: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    return audio.resample(x.mean(1), sr, SAMPLE_RATE)


def voicebench(out: Path):
    """The 1,064 dev items (OpenBookQA 142, MMSU 922); ids are "<file stem>/<row>", split by listen.splits."""
    (out / "qa").mkdir(parents=True, exist_ok=True)
    with open(out / "refs.jsonl", "w") as refs:
        for f in MCQ_FILES:
            for k, r in enumerate(rows(VB, f)):
                item = f"{f.split('-')[0]}/{k}"
                if split_of(item) == "dev":
                    name = item.replace("/", "_") + ".wav"
                    sf.write(out / "qa" / name, wav16(r["audio"]["bytes"]), SAMPLE_RATE)
                    ref = {"file": name, "reference": r["reference"].strip(), "prompt": r["prompt"]}
                    refs.write(json.dumps(ref) + "\n")


def pack(out: Path, libritts: Path | None = None):
    """qa/: obqa_000..039 (OpenBookQA rows 0-39); asr/: the listed LibriTTS-R clips; one refs.jsonl for both."""
    for sub in ("qa", "asr"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(out / "refs.jsonl", "w") as refs:
        for k, r in enumerate(rows(VB, MCQ_FILES[0])[:40]):
            sf.write(out / "qa" / f"obqa_{k:03d}.wav", wav16(r["audio"]["bytes"]), SAMPLE_RATE)
            refs.write(json.dumps({"file": f"obqa_{k:03d}.wav", "reference": r["reference"].strip(),
                                   "prompt": r["prompt"]}) + "\n")
        if libritts:
            found = {f.name: f for f in Path(libritts).rglob("*.wav")}
            for line in open(Path(__file__).with_name("librittsr50.jsonl")):
                r = json.loads(line)
                sf.write(out / "asr" / r["file"], audio.load(found[r["file"]]), SAMPLE_RATE)
                refs.write(line)


def scored(out: Path, name: str):
    """Every item of a VoiceBench subset scored by its own rules (bbh, ifeval); refs keep each row's fields."""
    (out / "qa").mkdir(parents=True, exist_ok=True)
    with open(out / "refs.jsonl", "w") as refs:
        for f in sorted(x for x in list_repo_files(VB, repo_type="dataset") if x.startswith(name + "/")):
            for r in rows(VB, f):
                a = r.pop("audio")
                r["file"] = (r.get("id") or f"{name}_{r['key']}") + ".wav"
                sf.write(out / "qa" / r["file"], wav16(a["bytes"]), SAMPLE_RATE)
                refs.write(json.dumps(r) + "\n")


def smartturn(out: Path, n_calib: int = 200, n_test: int = 500, seed: int = 0):
    """Real English clips, half complete and half incomplete, split into a calibration and a held-out set."""
    pool = []
    for shard in range(10):
        for r in rows("pipecat-ai/smart-turn-data-v3.2-test", f"data/train-{shard:05d}-of-00010.parquet"):
            if r["language"] == "eng" and not r["synthetic"]:
                pool.append(r)
        if sum(r["endpoint_bool"] for r in pool) >= n_calib + n_test and len(pool) >= 2 * (n_calib + n_test):
            break
    rng = np.random.default_rng(seed)
    by = {b: [r for r in pool if r["endpoint_bool"] == b] for b in (True, False)}
    for b in by:
        rng.shuffle(by[b])
    for name, lo, hi in (("calib", 0, n_calib // 2), ("test", n_calib // 2, (n_calib + n_test) // 2)):
        (out / name).mkdir(parents=True, exist_ok=True)
        with open(out / name / "labels.jsonl", "w") as labels:
            for r in by[True][lo:hi] + by[False][lo:hi]:
                sf.write(out / name / f"{r['id']}.wav", wav16(r["audio"]["bytes"]), SAMPLE_RATE)
                labels.write(json.dumps({"file": f"{r['id']}.wav", "complete": r["endpoint_bool"]}) + "\n")


def room_tone(x: np.ndarray, n: int) -> np.ndarray:
    """n samples of the clip's own background: its quietest 10 % of 20 ms frames, looped."""
    f = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
    quiet = f[np.argsort((f * f).mean(1))[: max(1, len(f) // 10)]].reshape(-1)
    return np.resize(quiet, n).astype(np.float32)


def speech_end(x: np.ndarray) -> float:
    """Seconds to the end of the last non-silent 20 ms frame (acoustic, as the endpointer sees it)."""
    s = Silence()
    s.start()
    s.push(x)
    loud = [i for i in range(len(s.db)) if not s.silent(i)]
    return (loud[-1] + 1) * FRAME / SAMPLE_RATE if loud else 0.0


def pauses(out: Path, seed: int = 0, tail: float = 2.0):
    """open/: 240 open-ended questions (VoiceBench sd-qa USA 160 + commoneval 80), each with one pause of 200-1000 ms
    inserted between two words (recognizer timestamps) and filled with room tone, + `tail` s of room tone.
    mcq/: 100 VoiceBench multiple-choice dev clips with the same tail, unchanged otherwise (the "stem trap")."""
    rng, asr = np.random.default_rng(seed), Parakeet()
    clips = [(f"sdqa_{k}", r) for k, r in enumerate(rows(VB, "sd-qa/usa-00000-of-00001.parquet")[:160])]
    clips += [(f"commoneval_{k}", r) for k, r in enumerate(rows(VB, "commoneval/test-00000-of-00001.parquet")[:80])]
    for sub in ("open", "mcq"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(out / "open" / "labels.jsonl", "w") as labels:
        for k, (name, r) in enumerate(clips):
            x = wav16(r["audio"]["bytes"])
            words = asr(x)
            d = (200, 400, 600, 800, 1000)[k % 5]
            if len(words) >= 3:  # a boundary with at least one word on each side
                j = int(rng.integers(0, len(words) - 2))
                at = int(SAMPLE_RATE * (words[j][2] + words[j + 1][1]) / 2)
                x = np.concatenate([x[:at], room_tone(x, d * SAMPLE_RATE // 1000), x[at:]])
            else:
                at, d = None, 0
            end = speech_end(x)
            x = np.concatenate([x, room_tone(x, int(tail * SAMPLE_RATE))])
            sf.write(out / "open" / f"{name}.wav", x, SAMPLE_RATE)
            labels.write(json.dumps({"file": f"{name}.wav", "pause_s": None if at is None else at / SAMPLE_RATE,
                                     "pause_ms": d, "speech_end": end, "prompt": r["prompt"]}) + "\n")
    refs = [json.loads(line) for line in open(out.parent / "vb_dev" / "refs.jsonl")][:100]
    with open(out / "mcq" / "labels.jsonl", "w") as labels:
        for r in refs:
            x = audio.load(out.parent / "vb_dev" / "qa" / r["file"])
            end = speech_end(x)
            sf.write(out / "mcq" / r["file"], np.concatenate([x, room_tone(x, int(tail * SAMPLE_RATE))]), SAMPLE_RATE)
            labels.write(json.dumps({"file": r["file"], "pause_s": None, "pause_ms": 0, "speech_end": end}) + "\n")


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd in ("bbh", "ifeval"):
        scored(out, cmd)
    elif cmd == "pack":
        pack(out, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        {"voicebench": voicebench, "smartturn": smartturn, "pauses": pauses}[cmd](out)
