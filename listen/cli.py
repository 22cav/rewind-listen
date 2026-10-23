"""`./run.sh <command>`: the commands on the Mac.

  ./run.sh setup                 install the environment and download the recognizer and the LLM (once)
  ./run.sh talk                  speak; the answer starts when you have finished (no button)
  ./run.sh ask FILE...           answer the spoken question in each recording (files or folders)
  ./run.sh transcribe FILE...    write down what is said
  ./run.sh help                  this text

Options: --llm any mlx-lm chat model (default Qwen3-1.7B, 4-bit), --asr a parakeet-mlx model.
"""
import argparse
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listen import audio  # noqa: E402
from listen.stream import DEFAULT_ASR, DEFAULT_LLM, FRAME, SAMPLE_RATE, Listener, Parakeet  # noqa: E402


def answer(lis: Listener):
    pieces = lis.finish()
    first = next(pieces, "")  # finish() reads the end of the transcript before the first piece
    print(f"> {lis.transcript}\n\n{first}", end="", flush=True)
    for piece in pieces:
        print(piece, end="", flush=True)
    print("\n")


def setup(args):
    from huggingface_hub import snapshot_download

    for repo in (args.asr, args.llm):
        if not Path(repo).is_dir():
            print(f"downloading {repo} ...")
            snapshot_download(repo)
    print("ready: try  ./run.sh talk  or  ./run.sh ask samples/question.wav")


def ask(args):
    lis = Listener(args.llm, Parakeet(args.asr))
    for path in audio.expand(args.files):
        x = audio.load(path)
        lis.start()
        for i in range(0, len(x), FRAME):  # as if spoken: the LLM reads the transcript while it grows
            lis.feed(x[i : i + FRAME])
        print(f"== {path.name}")
        answer(lis)


def talk(args):
    import sounddevice as sd

    lis, frames = Listener(args.llm, Parakeet(args.asr)), queue.Queue()
    print("speak (Ctrl-C to quit)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME, dtype="float32",
                        callback=lambda data, n, t, status: frames.put(data[:, 0].copy())):
        while True:
            lis.start()
            while not lis.feed(frames.get()):  # each 20 ms frame; True once the turn has ended
                pass
            answer(lis)
            while not frames.empty():  # drop what was heard while answering
                frames.get()


def transcribe(args):
    asr = Parakeet(args.asr)
    for path in audio.expand(args.files):
        print(f"{path.name}: {' '.join(w for w, _, _ in asr(audio.load(path)))}")


def main():
    ap = argparse.ArgumentParser(prog="./run.sh", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("command", choices=["setup", "talk", "ask", "transcribe", "help"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--llm", default=DEFAULT_LLM)
    ap.add_argument("--asr", default=DEFAULT_ASR)
    args = ap.parse_args()
    if args.command == "help":
        print(__doc__)
    elif args.command in ("ask", "transcribe") and not args.files:
        sys.exit(f"usage: ./run.sh {args.command} FILE...")
    else:
        {"setup": setup, "talk": talk, "ask": ask, "transcribe": transcribe}[args.command](args)


if __name__ == "__main__":
    main()
