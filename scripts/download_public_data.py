"""Download verified public noisy images or public evaluation targets.

The JSONL manifest contains name, Drive id and size from a verified public folder
listing. Existing images are verified and reused. Inference has no network code.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, expected_size: int) -> str:
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected file size: {path.name}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (992, 992):
            raise ValueError(f"Unexpected image format: {path.name}")
        image.verify()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(entry: dict, directory: Path) -> dict:
    target = directory / entry["name"]
    if target.exists():
        digest = verify(target, entry["size"])
        return {**entry, "sha256": digest, "reused": True}
    url = ("https://drive.usercontent.google.com/download?id=" + entry["id"]
           + "&export=download&confirm=t")
    for attempt in range(4):
        partial = None
        try:
            with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                dir=directory, prefix=entry["name"] + "-", suffix=".part", delete=False
            ) as stream:
                partial = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            digest = verify(partial, entry["size"])
            if target.exists():
                raise FileExistsError(f"Destination appeared during download: {target}")
            partial.rename(target)
            return {**entry, "sha256": digest, "reused": False}
        except (OSError, ValueError):
            if partial is not None:
                partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Download did not complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("noisy", "ground_truth"), default="noisy")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--split", choices=("public", "submissions"), default="public")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = ROOT / f"scripts/runs/public_{args.kind}_manifest.jsonl"
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12.")
    entries = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    suffix = "_noise" if args.kind == "noisy" else ""
    if args.split == "submissions" and args.kind != "noisy":
        parser.error("Submission ground truth is hidden; only noisy images may be downloaded.")
    ids = range(1, 461) if args.split == "public" else range(461, 481)
    expected = {f"{index:03d}{suffix}.png" for index in ids}
    if len(entries) != len(expected) or {entry["name"] for entry in entries} != expected:
        raise ValueError(f"Manifest must contain exactly the expected {args.split} {args.kind} images.")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", entry["id"]) for entry in entries):
        raise ValueError("Invalid Drive file id in manifest.")
    directory = ROOT / "competition_data" / args.split / args.kind
    directory.mkdir(parents=True, exist_ok=True)
    print(f"{args.split} {args.kind} | {len(entries)} images | {args.workers} download workers", flush=True)
    results, failures = [], []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download, entry, directory): entry for entry in entries}
        for future in as_completed(pending):
            entry = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"name": entry["name"], "error": str(error)})
                print(f"Failed: {entry['name']}: {error}", flush=True)
            count = len(results) + len(failures)
            if count % 20 == 0 or count == len(entries):
                print(f"[{count:03d}/{len(entries)}] {len(results)} verified | {time.monotonic()-started:.0f}s", flush=True)
    report = args.manifest.with_name(f"{args.split}_{args.kind}_download.json")
    report.write_text(json.dumps({"images": sorted(results, key=lambda row: row["name"]),
                                  "failures": failures}, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
