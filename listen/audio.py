"""Audio files in: any file soundfile reads (wav, flac, ogg, mp3, aiff), plus m4a/aac/caf (Voice Memos,
iPhone) on macOS through the system's afconvert; mixed to mono and resampled (polyphase, scipy) to the recognizer's
16 kHz."""
import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AFCONVERT_EXTENSIONS = (".m4a", ".aac", ".caf") if shutil.which("afconvert") else ()
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff") + AFCONVERT_EXTENSIONS


def _read_afconvert(path):
    """Decode with macOS afconvert to a temporary 32-bit float wav (at the file's own rate)."""
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEF32", str(path), str(wav)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise ValueError(f"{path}: afconvert failed: {p.stderr.strip() or p.stdout.strip()}")
        return sf.read(str(wav), dtype="float32", always_2d=True)


def load(path, sample_rate: int = 16_000) -> np.ndarray:
    """-> float32 mono waveform at sample_rate."""
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        if Path(path).suffix.lower() not in AFCONVERT_EXTENSIONS:
            supported = ", ".join(AUDIO_EXTENSIONS)
            raise ValueError(f"{path}: cannot read ({e.error_string}); supported: {supported}") from None
        x, sr = _read_afconvert(path)
    return resample(x.mean(axis=1), sr, sample_rate)


def resample(x: np.ndarray, sr: int, sample_rate: int = 16_000) -> np.ndarray:
    """Polyphase resampling of a mono float32 waveform."""
    if sr == sample_rate:
        return x.astype(np.float32)
    g = gcd(sr, sample_rate)
    return resample_poly(x, sample_rate // g, sr // g).astype(np.float32)


def expand(inputs, extensions=AUDIO_EXTENSIONS) -> list[Path]:
    """Files as given, and every matching file under directories (sorted)."""
    out = []
    for p in map(Path, inputs):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in extensions)
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out
