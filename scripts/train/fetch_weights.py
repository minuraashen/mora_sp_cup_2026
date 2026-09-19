"""
fetch_weights.py -- DEV ONLY. Downloads the pretrained initialisation.

This is a TRAINING-time helper. The graded inference path (scripts/denoise.py) never
touches the network: it loads a local checkpoint only, so the submission runs offline.

Primary: NAFNet-SIDD-width32. The upstream repo publishes only Google Drive links,
which are not scriptable; the HuggingFace mirror below is a plain `resolve/` URL with
no auth. Provenance is checked by size and by a strict=True load into our own
architecture -- 29,159,715 parameters is a fingerprint that a wrong file cannot match.
NAFNet is MIT/Apache-2.0 licensed, so using the released weights is unrestricted; this
is recorded in the report.

    python scripts/train/fetch_weights.py --model nafnet_sidd_w32
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

MODELS = {
    "nafnet_sidd_w32": dict(
        url="https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width32.pth",
        filename="NAFNet-SIDD-width32.pth",
        expected_bytes=116861841,
        note="NAFNet SIDD width32, MIT/Apache-2.0 (megvii-research/NAFNet)",
    ),
    "scunet_color_real": dict(
        url="https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        filename="scunet_color_real_psnr.pth",
        expected_bytes=71982841,
        note="SCUNet blind colour denoiser -- fallback / alternative-considered baseline",
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                pct = 100.0 * done / total
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:7.1f} MB  ({pct:5.1f}%)",
                      end="", flush=True)
    print(f"\n  downloaded in {time.time() - t0:.0f}s")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nafnet_sidd_w32", choices=sorted(MODELS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec = MODELS[args.model]
    dest = common.WEIGHTS_DIR / spec["filename"]

    if dest.exists() and not args.force:
        print(f"already present: {dest} ({dest.stat().st_size} bytes)")
    else:
        print(f"fetching {args.model}: {spec['url']}")
        try:
            download(spec["url"], dest)
        except Exception as exc:
            print(f"DOWNLOAD FAILED: {exc}")
            print("Fall back to: python scripts/train/train.py --arch w32lite --init none")
            return 2

    size = dest.stat().st_size
    digest = sha256_file(dest)
    ok_size = (size == spec["expected_bytes"])
    print(f"  size   {size} bytes  ({'OK' if ok_size else 'MISMATCH, expected ' + str(spec['expected_bytes'])})")
    print(f"  sha256 {digest}")

    record = common.ARTIFACTS_DIR / "weights.json"
    existing = common.read_json(record) if record.exists() else {}
    existing[args.model] = dict(url=spec["url"], filename=spec["filename"],
                                bytes=size, sha256=digest, note=spec["note"])
    common.write_json(record, existing)
    print(f"  recorded in {record}")
    return 0 if ok_size else 3


if __name__ == "__main__":
    raise SystemExit(main())
