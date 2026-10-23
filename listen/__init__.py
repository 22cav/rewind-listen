"""An unmodified on-device LLM that reads a spoken question while it is being spoken (MLX, no PyTorch).

  stream  the streaming listener: recognizer passes, the transcript read into the LLM's KV cache, end of turn
  llm     any mlx-lm chat model: load, read token ids into the cache, chat template split, greedy decoding
  audio   audio files in, mono 16 kHz
  mcq     VoiceBench's multiple-choice answer extraction (for scoring)
  splits  the fixed VoiceBench development / test split
"""
