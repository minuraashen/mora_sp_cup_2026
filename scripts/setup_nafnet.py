"""Download the official SIDD-width32 checkpoint once; inference stays offline."""

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import urlopen

WEIGHTS_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ&export=download&confirm=t"
)
DEFAULT_PATH = Path(__file__).resolve().parent / "weights" / "NAFNet-SIDD-width32.pth"
EXPECTED_SHA256 = "89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        with args.output.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != EXPECTED_SHA256:
            raise ValueError("Existing checkpoint differs from the verified official download. "
                             "Use --weights to run a custom checkpoint; setup will not replace it.")
        print(f"Already present: {args.output}\nSHA-256: {digest}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    print("Downloading official NAFNet-SIDD-width32 weights (about 117 MB)...", flush=True)
    try:
        with urlopen(WEIGHTS_URL, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=args.output.parent, prefix=args.output.stem + "-", suffix=".part", delete=False
        ) as stream:
            partial = Path(stream.name)
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Google Drive returned a web page instead of weights.")
            received = 0
            last_update = time.monotonic()
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if time.monotonic() - last_update >= 10:
                    print(f"  {received / 1_000_000:.1f} MB downloaded", flush=True)
                    last_update = time.monotonic()
        with partial.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != EXPECTED_SHA256:
            raise ValueError("Checkpoint SHA-256 mismatch; download was not installed.")
        # Load tensors using PyTorch's restricted unpickler.
        import torch
        payload = torch.load(partial, map_location="cpu", weights_only=True)
        if "params" not in payload or "intro.weight" not in payload["params"]:
            raise ValueError("Downloaded file is not the expected NAFNet checkpoint.")
        if args.output.exists():
            raise FileExistsError(f"Destination appeared during download: {args.output}")
        partial.rename(args.output)
    except BaseException:
        if partial is not None:
            partial.unlink(missing_ok=True)
        raise
    with args.output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"Saved: {args.output}\nSHA-256: {digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
