"""Streaming a spoken turn into an unmodified LLM.

At each pause (and after 4 s of speech without one) the recognizer transcribes the current audio window; the
transcript is tokenized and the LLM's KV cache is rewound to the longest prefix it shares with the cached tokens, then
only the changed tail is read. Causal attention makes the cache always equal to reading the current transcript from
scratch.
The LLM's probability that the next character closes the sentence (q_end) sets how much trailing silence ends the turn.

  from listen.stream import Listener
  lis = Listener()                                  # Parakeet 110M + Qwen3-1.7B 4-bit, or any mlx-lm chat model
  lis.start()
  for frame in frames_of_20ms_at_16khz:
      if lis.feed(frame): break                     # True when the turn has ended
  for piece in lis.finish(): print(piece, end="")
"""
import re
import time

import mlx.core as mx
import numpy as np

from . import llm

DEFAULT_ASR = "mlx-community/parakeet-tdt_ctc-110m"
DEFAULT_LLM = "mlx-community/Qwen3-1.7B-4bit"
SYSTEM = "You are a helpful assistant."
SAMPLE_RATE = 16_000
FRAME = 320  # 20 ms


class Silence:
    """Trailing silence from 20 ms frame energies. A frame is silent below noise floor + 10 dB; the floor is the 10th
    percentile of the turn's frames, seeded low and capped at -42 dBFS so speech can never become the floor."""

    def __init__(self, ceiling=-42.0, seed=-55.0, margin=10.0):
        self.ceiling, self.seed, self.margin = ceiling, seed, margin

    def start(self):
        self.db, self.floor, self.rest = [], [], np.zeros(0, np.float32)

    def push(self, samples: np.ndarray):
        x = np.concatenate([self.rest, samples])
        n = len(x) // FRAME
        self.rest = x[n * FRAME:]
        for f in x[: n * FRAME].reshape(n, FRAME):
            self.db.append(max(-90.0, float(10 * np.log10(np.mean(f * f) + 1e-12))))
            self.floor.append(min(float(np.percentile([self.seed] * 15 + self.db, 10)), self.ceiling))

    def silent(self, i: int) -> bool:
        return self.db[i] < self.floor[i] + self.margin

    def trailing_ms(self) -> float:
        n = 0
        while n < len(self.db) and self.silent(len(self.db) - 1 - n):
            n += 1
        return 20.0 * n

    def pause_cut(self, lo: int, hi: int, min_frames: int = 5, latest: bool = True) -> int | None:
        """Frame index in the middle of a silent run (>= min_frames) inside [lo, hi): the latest one, or the longest;
        None if there is none."""
        best, run = None, 0
        for i in range(lo, hi):
            run = run + 1 if self.silent(i) else 0
            if run >= min_frames and (best is None or latest or run > best[1]):
                best = (i - run + 1, run)
        return None if best is None else best[0] + best[1] // 2


def words_of(tokens) -> list[tuple[str, float, float]]:
    """Recognizer subword tokens (a leading space starts a word) -> [(word, start s, end s)]."""
    out = []
    for t in tokens:
        if t.text.startswith(" ") or not out:
            out.append([t.text.strip(), t.start, t.end])
        else:
            out[-1][0] += t.text
            out[-1][2] = t.end
    return [tuple(w) for w in out if w[0]]


class Parakeet:
    """An offline parakeet-mlx model, run on the current audio window."""

    def __init__(self, model_id: str = DEFAULT_ASR):
        from parakeet_mlx import from_pretrained

        self.model = from_pretrained(model_id)

    def __call__(self, audio: np.ndarray) -> list[tuple[str, float, float]]:
        from parakeet_mlx.audio import get_logmel

        if len(audio) < 2 * self.model.preprocessor_config.hop_length:
            return []
        mel = get_logmel(mx.array(audio), self.model.preprocessor_config)
        return words_of(self.model.generate(mel)[0].tokens)


def closing_ids(tok, newline: bool = False) -> mx.array:
    """Tokens whose first character closes a sentence (".", "?", "!", not followed by a letter or digit, so not
    ".com" or ".5"), plus the end-of-turn tokens. Summing their probabilities marginalizes BPE merges like "?\\n\\n".
    With newline=True, tokens starting with a newline count too (an ablation)."""
    first = ".?!\n" if newline else ".?!"
    ids = [i for i in range(tok.vocab_size) if (s := tok.decode([i]))[:1] in first and not s[1:2].isalnum()]
    return mx.array(sorted(set(ids) | llm.end_ids(tok)))


def lcp(a: list[int], b: list[int]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


class Listener:
    """Recognizer + LLM over one turn. feed() takes 16 kHz audio (any length) and returns True when the turn ends;
    finish() reads the chat suffix and yields the answer. policy "lcp" reads every transcript at once (rewinding what
    changed); "la2" (LocalAgreement-2, the baseline) reads only tokens two consecutive passes agree on."""

    def __init__(self, llm_id: str = DEFAULT_LLM, asr=None, system: str = SYSTEM, policy: str = "lcp",
                 step: float = 4.0, window: float = 8.0, cap: float = 14.0, keep: float = 3.0,
                 tau=(300.0, 1200.0), gamma: float = 1.0,
                 onset_ms: float = 120.0, model=None, tok=None):
        self.asr = asr or Parakeet()
        self.model, self.tok = (model, tok) if model is not None else llm.load(llm_id)
        self.before, self.after = llm.chat_pieces(self.tok, system, "")
        self.closing, self.closing_nl = closing_ids(self.tok), closing_ids(self.tok, newline=True)
        self.policy, self.step_n, self.window, self.cap, self.keep = policy, int(step * SAMPLE_RATE), window, cap, keep
        self.tau, self.gamma, self.onset_ms = tau, gamma, onset_ms
        self.silence = Silence()

    def start(self):
        self.cache = llm.new_cache(self.model)
        llm.prefill(self.model, llm.embed(self.model, self.before), self.cache)  # the chat prefix, read once
        self.cached, self.prev = [], []    # token ids in the cache after the prefix; the previous pass's ids
        self.buf, self.t0, self.n = np.zeros(0, np.float32), 0, 0  # window audio, its start and total (samples)
        self.frozen, self.raw = "", ""     # text before the window; the last pass's full transcript
        self.since, self.onset, self.q_end, self.q_nl = 0, False, 0.0, 0.0
        self.heard = False                 # a non-silent frame since the last pass
        self.silence.start()
        self.steps, self.cuts, self.forced = [], 0, 0

    def tau_wait(self) -> float:
        lo, hi = self.tau
        return hi - (hi - lo) * self.q_end**self.gamma

    def feed(self, samples: np.ndarray) -> bool:
        samples = samples.astype(np.float32)
        seen = len(self.silence.db)
        self.silence.push(samples)
        self.heard = self.heard or not all(self.silence.silent(i) for i in range(seen, len(self.silence.db)))
        self.buf = np.concatenate([self.buf, samples])
        self.n, self.since = self.n + len(samples), self.since + len(samples)
        quiet = self.silence.trailing_ms()
        if quiet < self.onset_ms:
            self.onset = False
        # silence onset (a pause), or a step after `step` s with none (skipped when only silence came since the last
        # pass: nothing new to read; LocalAgreement still needs it to confirm the held-back words)
        step = self.since >= self.step_n and (self.heard or self.policy == "la2")
        if step or (quiet >= self.onset_ms and not self.onset):
            self.onset = self.onset or quiet >= self.onset_ms
            self.pass_()
        return bool(self.cached) and quiet >= self.tau_wait()

    def pass_(self, final: bool = False):
        """Transcribe the window, cut it at a pause when too long, and sync the cache with the transcript."""
        t0 = time.perf_counter()
        words = [(w, s + self.t0 / SAMPLE_RATE, e + self.t0 / SAMPLE_RATE) for w, s, e in self.asr(self.buf)]
        t1 = time.perf_counter()
        if not final and len(self.buf) > self.window * SAMPLE_RATE:
            self.cut(words)
            words = [w for w in words if w[1] >= self.t0 / SAMPLE_RATE]
        self.raw = " ".join([self.frozen] + [w[0] for w in words]).strip()
        text = self.raw if final else re.sub(r"[.?!]+$", "", self.raw)  # a slice always "ends": drop that period
        ids = self.tok.encode(text, add_special_tokens=False) if text else []
        target = ids if final or self.policy == "lcp" else self.agreed(ids)
        k, r, adv = self.sync(target)
        self.prev, self.since, self.heard = ids, 0, False
        self.steps.append({"t": round(self.n / SAMPLE_RATE, 3), "n": len(ids), "k": k, "r": r, "adv": adv,
                           "q": round(self.q_end, 4), "q_nl": round(self.q_nl, 4), "sil": self.silence.trailing_ms(),
                           "tau": round(self.tau_wait(), 1), "asr_ms": round(1e3 * (t1 - t0), 2),
                           "sync_ms": round(1e3 * (time.perf_counter() - t1), 2), "final": final})

    def cut(self, words):
        """Freeze the words before the latest pause (>= 100 ms, not in the last `keep` s) and restart the window
        there; with no pause, wait until the `cap` and cut at the quietest 100 ms."""
        f0 = self.t0 // FRAME
        hi = len(self.silence.db) - int(self.keep * SAMPLE_RATE) // FRAME
        c = self.silence.pause_cut(f0, hi)
        if c is None:
            if len(self.buf) <= self.cap * SAMPLE_RATE:
                return
            db = np.convolve(self.silence.db[f0:hi], np.ones(5), "valid")
            c, self.forced = f0 + int(np.argmin(db)) + 2, self.forced + 1
        t_cut = c * FRAME  # inside a pause: a word that starts before it was spoken before it (end times lag)
        self.frozen = " ".join([self.frozen] + [w[0] for w in words if w[1] < t_cut / SAMPLE_RATE]).strip()
        self.buf, self.t0, self.cuts = self.buf[t_cut - self.t0:], t_cut, self.cuts + 1

    def agreed(self, ids: list[int]) -> list[int]:
        """LocalAgreement-2 on tokens: the cache grows by the prefix the last two passes share, never shrinks."""
        k = lcp(self.prev, ids)
        return ids[:k] if k > len(self.cached) and ids[: len(self.cached)] == self.cached else self.cached

    def sync(self, target: list[int]) -> tuple[int, int, int]:
        """Rewind the cache to the longest common prefix with target, read the new tail, update q_end. The cache never
        shrinks below the chat prefix: only tokens in self.cached are ever trimmed."""
        k = lcp(self.cached, target)
        r = len(self.cached) - k
        tail = target[k:]
        if not tail and r and k:  # a pure rewind: re-read the last kept token so q_end is current
            k, tail = k - 1, target[k - 1 : k]
        if len(self.cached) > k:
            from mlx_lm.models.cache import trim_prompt_cache

            trim_prompt_cache(self.cache, len(self.cached) - k)
        self.cached = self.cached[:k] + tail
        if tail:
            p = mx.softmax(llm.forward(self.model, llm.embed(self.model, tail), self.cache, last_only=True)[0, -1])
            self.q_end, self.q_nl = p[self.closing].sum().item(), p[self.closing_nl].sum().item()
        elif not self.cached:
            self.q_end = self.q_nl = 0.0
        return k, r, len(tail)

    def finish(self, clock: dict | None = None, max_tokens: int = 512):
        """End of turn: a final pass with punctuation (only if speech arrived since the last pass), the chat suffix,
        then the answer, greedily. clock gets "synced" (transcript read) and "token1" (first token chosen)."""
        new = range(max(0, len(self.silence.db) - self.since // FRAME), len(self.silence.db))
        if self.since and not all(self.silence.silent(i) for i in new):
            self.pass_(final=True)
        else:  # the silence-onset pass saw all the speech: restore its punctuation only
            self.sync(self.tok.encode(self.raw, add_special_tokens=False) if self.raw else [])
        if clock is not None:
            clock["synced"] = time.perf_counter()
        self.logits = llm.forward(self.model, llm.embed(self.model, self.after), self.cache, last_only=True)
        tok = int(mx.argmax(self.logits[0, -1]).item())
        if clock is not None:
            clock["token1"] = time.perf_counter()
        yield from llm.decode(self.model, self.tok, tok, self.cache, max_tokens)

    @property
    def transcript(self) -> str:
        return self.raw

    def log(self) -> dict:
        """Everything the analyses need: per-step records and the per-frame energy log."""
        return {"steps": self.steps, "db": [round(x, 1) for x in self.silence.db],
                "floor": [round(x, 1) for x in self.silence.floor], "cuts": self.cuts, "forced_cuts": self.forced}
