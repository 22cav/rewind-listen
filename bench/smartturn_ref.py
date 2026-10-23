"""Smart Turn v3.2's own score on the Smart Turn clips, as a reference for q_end (analyze.py completion): its
completion probability on each clip, preprocessed as in pipecat-ai/smart-turn inference.py (last 8 s, zero-padded at
the start, Whisper features). Needs onnxruntime + transformers (no PyTorch): use a separate environment.

  $STPY bench/smartturn_ref.py data/smartturn/test
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

SR, N = 16_000, 8 * 16_000


def auroc(pos, neg) -> float:  # as in analyze.py
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def main(d: Path):
    session = ort.InferenceSession(hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"))
    fe = WhisperFeatureExtractor(chunk_length=8)
    pos, neg = [], []
    for r in map(json.loads, open(d / "labels.jsonl")):
        x, sr = sf.read(d / r["file"], dtype="float32")
        assert sr == SR
        x = x[-N:] if len(x) > N else np.pad(x, (N - len(x), 0))
        f = fe(x, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=N, truncation=True,
               do_normalize=True).input_features.astype(np.float32)
        (pos if r["complete"] else neg).append(session.run(None, {"input_features": f})[0].item())
    print(f"{d}: Smart Turn v3.2: AUROC {auroc(pos, neg):.3f} | median complete {np.median(pos):.3f}, "
          f"incomplete {np.median(neg):.3f} (n {len(pos)} + {len(neg)})")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
