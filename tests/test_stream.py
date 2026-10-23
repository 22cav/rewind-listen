"""listen/stream.py: the cache equals an offline read after any rewind, the guards, the acoustic silence measure, the
pause cut, the frame-resolution stop and the completion set. A tiny random Qwen3 and a byte tokenizer: no network."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import llm, splits, stream  # noqa: E402
from listen.stream import FRAME, SAMPLE_RATE  # noqa: E402

QWEN3_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"  # the pinned Qwen/Qwen3-1.7B snapshot


class Tok:
    """Byte-level stand-in tokenizer (ids < 256, the tiny LLM's vocabulary) with a fixed chat template."""
    eos_token_id, eos_token_ids, unk_token_id, vocab_size = 4, set(), None, 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode(errors="replace")

    def convert_tokens_to_ids(self, token):
        return None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}</>" for m in messages) + "<assistant>"


def tiny_llm():
    import mlx_lm.models.qwen3 as qwen3

    mx.random.seed(0)
    return qwen3.Model(qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
                                       num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=256,
                                       tie_word_embeddings=True, head_dim=16, max_position_embeddings=512,
                                       rope_theta=1e6))


class Script:
    """A fake recognizer: returns the next scripted word list on each call."""

    def __init__(self, passes=None):
        self.passes = list(passes or [])

    def __call__(self, audio):
        return self.passes.pop(0) if len(self.passes) > 1 else self.passes[0] if self.passes else []


def listener(asr=None, **kw):
    lis = stream.Listener(asr=asr or Script(), model=tiny_llm(), tok=Tok(), system="sys", **kw)
    lis.start()
    return lis


def offline(lis, text, suffix=False):
    """Logits at the last position when the chat prefix + text (+ suffix) is read from scratch."""
    ids = lis.before + lis.tok.encode(text) + (lis.after if suffix else [])
    return llm.forward(lis.model, llm.embed(lis.model, ids), llm.new_cache(lis.model), last_only=True)


def q_of(lis, logits):
    return mx.softmax(logits[0, -1])[lis.closing].sum().item()


def test_cache_equals_offline_read_after_rewinds():
    lis = listener()
    for text in ["the cat", "the cats sat", "the cat sat on", "the dog", "the dog ran far"]:
        lis.sync(lis.tok.encode(text))
        assert abs(lis.q_end - q_of(lis, offline(lis, text))) < 1e-5
    lis.raw = "the dog ran far"
    list(lis.finish(max_tokens=1))
    assert mx.allclose(lis.logits, offline(lis, "the dog ran far", suffix=True), atol=1e-4).item()


def test_pure_rewind_rereads_last_token():
    lis = listener()
    lis.sync(lis.tok.encode("hello there"))
    k, r, adv = lis.sync(lis.tok.encode("hello"))
    assert (k, r, adv) == (4, 6, 1)
    assert abs(lis.q_end - q_of(lis, offline(lis, "hello"))) < 1e-5


def test_cache_never_shrinks_below_the_chat_prefix():
    lis = listener()
    lis.sync(lis.tok.encode("noise"))
    lis.sync([])
    assert lis.cached == [] and lis.q_end == 0.0
    assert all(c.offset == len(lis.before) for c in lis.cache)


def test_trailing_punctuation_dropped_until_the_final_pass():
    lis = listener(Script([[("Hello", 0.0, 0.3), ("world.", 0.4, 0.8)]]))
    lis.feed(np.zeros(FRAME, np.float32))
    lis.pass_()
    assert lis.cached == lis.tok.encode("Hello world")
    lis.pass_(final=True)
    assert lis.cached == lis.tok.encode("Hello world.")


def tone(seconds, db=-20.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_acoustic_silence_from_the_waveform():
    rng = np.random.default_rng(0)
    for bed in (0.0, 10 ** (-50 / 20)):  # digital silence, then a noisy room at -50 dBFS
        s = stream.Silence()
        s.start()
        noise = lambda n, bed=bed: (bed * rng.standard_normal(n)).astype(np.float32)
        s.push(tone(1.0) + noise(SAMPLE_RATE))
        assert s.trailing_ms() == 0.0  # speaking
        s.push(noise(int(0.5 * SAMPLE_RATE)))
        assert abs(s.trailing_ms() - 500) <= 20


def test_noise_floor_cannot_start_inside_speech():
    s = stream.Silence()
    s.start()
    s.push(tone(2.0, db=-25.0))  # speech from the first frame, no leading silence
    assert max(s.floor) <= -42.0 and s.trailing_ms() == 0.0


def test_pause_cut_takes_the_latest_pause():
    s = stream.Silence()
    s.start()
    for part in (tone(1.0), np.zeros(4000), tone(1.0), np.zeros(3200), tone(1.0)):  # pauses of 250 and 200 ms
        s.push(np.asarray(part, np.float32))
    c = s.pause_cut(0, len(s.db))
    assert abs(c * FRAME / SAMPLE_RATE - 2.35) <= 0.04  # middle of the second (later) pause, 2.25-2.45 s
    assert s.pause_cut(0, 40) is None  # no pause in the first 0.8 s


def test_window_freezes_words_before_the_cut():
    words = [("one", 0.1, 0.5), ("two", 1.4, 1.8), ("three", 2.9, 3.3)]
    lis = listener(window=2.0, keep=0.5)
    lis.asr = lambda audio: [(w, s - lis.t0 / SAMPLE_RATE, e - lis.t0 / SAMPLE_RATE) for w, s, e in words
                             if s >= lis.t0 / SAMPLE_RATE]
    for part in (tone(0.6), np.zeros(8000), tone(0.8), np.zeros(8000), tone(0.6), np.zeros(800)):
        lis.feed(np.asarray(part, np.float32))
    lis.pass_()
    assert lis.cuts >= 1 and lis.frozen.startswith("one") and lis.raw == "one two three"


def test_stop_fires_on_the_crossing_frame_and_onset_pass_runs_once():
    lis = listener(Script([[("hi", 0.0, 0.5)]]), tau=(340.0, 340.0))
    lis.feed(tone(1.0))
    n_steps, fired = len(lis.steps), None
    for k in range(60):
        if lis.feed(np.zeros(FRAME, np.float32)):
            fired = k + 1
            break
    assert fired == 17  # 340 ms of silence = 17 frames, not the next step
    onset = [s for s in lis.steps[n_steps:] if 100 <= s["sil"] < 500]
    assert len(onset) == 1 and onset[0]["sil"] == 120.0


def test_no_step_pass_over_silence_alone():
    lis = listener(Script([[("hi", 0.0, 0.5)]]), tau=(5000.0, 5000.0))
    lis.feed(tone(1.0))
    n_steps = len(lis.steps)
    for _ in range(100):  # 2 s of silence: only the onset pass runs
        lis.feed(np.zeros(FRAME, np.float32))
    assert [s["sil"] for s in lis.steps[n_steps:]] == [120.0]


def test_closing_set_with_the_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=QWEN3_REVISION,
                                                         local_files_only=True)
    except Exception:
        pytest.skip("Qwen3 tokenizer not cached")
    ids = set(stream.closing_ids(tok).tolist())
    one = lambda s: tok.convert_tokens_to_ids(tok.tokenize(s))[0]
    assert {one("?"), one("?\n"), one(".\n\n"), tok.convert_tokens_to_ids("<|im_end|>")} <= ids
    assert tok.tokenize(".com") == [".com"] and one(".com") not in ids


def test_voicebench_dev_split():
    refs = Path(os.environ.get("DATA", Path(__file__).resolve().parents[1] / "data")) / "vb_dev/refs.jsonl"
    if not refs.exists():  # built by bench/data.py voicebench
        pytest.skip("VoiceBench dev pack not built")
    ids = [r["file"][:-4].replace("_", "/", 2) for r in map(json.loads, open(refs))]
    assert len(ids) == 1064 and all(splits.split_of(i) == "dev" for i in ids)
    assert sum(i.startswith("openbookqa") for i in ids) == 142
