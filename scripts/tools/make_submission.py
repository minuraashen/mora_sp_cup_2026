"""
make_submission.py -- build Voxel.zip containing exactly the 20 denoised PNGs.

Members are written in sorted order with a fixed timestamp and fixed compression, so
the zip file itself is byte-reproducible rather than merely containing the right
pixels. Nothing but the 20 images goes in: no directory prefix, no noisy inputs, no
OS metadata files.

    python scripts/tools/make_submission.py
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EXPECTED_IDS = [f"{i:03d}" for i in range(461, 481)]
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path,
                    default=REPO / "competition_data" / "submissions" / "denoised")
    ap.add_argument("--out", type=Path, default=REPO / "Voxel.zip")
    args = ap.parse_args()

    missing = [i for i in EXPECTED_IDS if not (args.pred_dir / f"{i}.png").exists()]
    if missing:
        print(f"ERROR: missing predictions: {missing}")
        return 1

    if args.out.exists():
        args.out.unlink()

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i in EXPECTED_IDS:
            src = args.pred_dir / f"{i}.png"
            info = zipfile.ZipInfo(filename=f"{i}.png", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, src.read_bytes())

    h = hashlib.sha256(args.out.read_bytes()).hexdigest()
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"  members ({len(names)}): {names[0]} .. {names[-1]}")
    print(f"  SHA-256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
