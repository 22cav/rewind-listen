"""The LLM side, with mlx-lm: load any chat model, read token ids into its KV cache, split the chat template around
the speaker's text, and decode greedily."""
import mlx.core as mx

END_TOKENS = ("<|im_end|>", "<|endoftext|>")  # Qwen chat end of turn and end of text: stop on either


def load(path_or_id: str):
    """-> (model, tokenizer) from an mlx-lm directory or Hugging Face id."""
    from mlx_lm import load as mlx_load

    return mlx_load(path_or_id)


def embed(model, ids) -> mx.array:
    """Token ids [T] -> input embeddings [1, T, width]."""
    return model.model.embed_tokens(mx.array(ids)[None])


def new_cache(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def hidden(model, embeds: mx.array, cache=None) -> mx.array:
    """Advance the transformer cache without projecting to the vocabulary."""
    emb = model.model.embed_tokens  # a quantized LLM's weight is packed uint32: take the float dtype from its scales
    h = embeds.astype(emb.scales.dtype if hasattr(emb, "scales") else emb.weight.dtype)
    return model.model(None, cache=cache, input_embeddings=h)


def prefill(model, embeds, cache):
    h = hidden(model, embeds, cache)
    mx.eval(h, [c.state for c in cache])


def forward(model, embeds: mx.array, cache=None, last_only: bool = False) -> mx.array:
    """Input vectors -> float32 vocabulary logits; optionally project only the final position."""
    h = hidden(model, embeds, cache)
    if last_only:
        h = h[:, -1:]
    inner = model.model
    if getattr(model, "args", None) is not None and getattr(model.args, "tie_word_embeddings", False):
        return inner.embed_tokens.as_linear(h).astype(mx.float32)
    return model.lm_head(h).astype(mx.float32)


def chat_text(tokenizer, system: str, user: str) -> str:
    """The model's chat template for (system, user) up to the assistant's turn, in non-thinking mode (Qwen3 adds an
    empty <think></think> block after the assistant header; templates without a thinking mode ignore the flag)."""
    return tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)


def chat_pieces(tokenizer, system: str, prompt: str) -> tuple[list[int], list[int]]:
    """Token ids of the chat template before and after the transcript."""
    marker = "\x00TEXT\x00"
    before, after = chat_text(tokenizer, system, f"{prompt} {marker}" if prompt else marker).split(marker)
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    return enc(before), enc(after)


def end_ids(tokenizer) -> set[int]:
    """Token ids that end the answer: the tokenizer's EOS set plus the Qwen end-of-turn / end-of-text tokens."""
    extra = getattr(tokenizer, "eos_token_ids", None) or set()  # a set (mlx-lm) or one int (some HF tokenizers)
    ids = {tokenizer.eos_token_id, *([extra] if isinstance(extra, int) else extra)}
    unk = getattr(tokenizer, "unk_token_id", None)
    ids |= {tokenizer.convert_tokens_to_ids(t) for t in END_TOKENS}  # a token lookup, not get_vocab (151k entries)
    return {i for i in ids if i is not None and i != unk}


def decode(model, tokenizer, tok: int, cache, max_tokens: int = 200):
    """Yield decoded text pieces, greedily, starting from an already selected first token, until end of turn."""
    eos = end_ids(tokenizer)
    for _ in range(max_tokens):
        if tok in eos:
            break
        yield tokenizer.decode([tok])
        logits = forward(model, embed(model, [tok]), cache)
        tok = int(mx.argmax(logits[0, -1]).item())


def greedy(model, tokenizer, first_logits: mx.array, cache, max_tokens: int = 200):
    """Yield decoded text pieces, greedily, from the logits of the last prompt position until end of turn."""
    yield from decode(model, tokenizer, int(mx.argmax(first_logits[0, -1]).item()), cache, max_tokens)
