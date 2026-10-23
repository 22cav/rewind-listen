"""Written VoiceBench prompts rewritten as error-free transcripts, for the gap analysis (cascade.py --text reads them):

  python bench/prompts.py flat REFS > flat.jsonl        line breaks removed
  python bench/prompts.py spoken REFS > spoken.jsonl    also numbers in words and bare option letters, as recognized
"""
import json
import re
import sys

from num2words import num2words


def say(n: str) -> str:
    """One number as spoken: four-digit years as years, decimals digit by digit."""
    n = n.replace(",", "")
    if "." in n:
        whole, frac = n.split(".", 1)
        return f"{num2words(int(whole or 0))} point {' '.join(num2words(int(d)) for d in frac)}"
    k = int(n)
    return num2words(k, to="year") if 1100 <= k <= 2099 and len(n) == 4 else num2words(k)


def flat(p: str) -> str:
    return " ".join(p.split())


def spoken(p: str) -> str:
    p = re.sub(r"\n([A-D])\. ", r" \1 ", p)  # "\nA. x" -> " A x"
    p = p.replace("following: A, B, C, or D.", "following A, B, C, or D")
    p = re.sub(r"\$(\d[\d,]*)\.(\d\d)\b",
               lambda m: f"{say(m[1])} dollars" + (f" and {say(m[2])} cents" if m[2] != "00" else ""), p)
    p = re.sub(r"\$(\d[\d,]*(?:\.\d+)?)", lambda m: f"{say(m[1])} dollars", p)
    p = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", lambda m: f"{say(m[1])} per cent", p)
    p = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: say(m[0]), p)
    return flat(p.replace("-", " "))


if __name__ == "__main__":
    form = {"flat": flat, "spoken": spoken}[sys.argv[1]]
    for r in map(json.loads, open(sys.argv[2])):
        print(json.dumps({"file": r["file"], "prompt": form(r["prompt"])}))
