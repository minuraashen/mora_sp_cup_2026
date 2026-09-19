"""Package the denoised preliminary-round outputs as Voxel.zip.

    python scripts/make_submission.py

Validates the 20 outputs before zipping: correct filenames, PNG, RGB, and the
same dimensions as the corresponding noisy input. The archive is flat -- 20
files, no directory nesting.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image

from ncw.paths import REPO_ROOT, SUBMISSION_DENOISED, SUBMISSION_NOISY

FIRST_ID = 461
LAST_ID = 480


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--denoised_dir", type=Path, default=SUBMISSION_DENOISED)
    parser.add_argument("--noise_dir", type=Path, default=SUBMISSION_NOISY)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "Voxel.zip")
    args = parser.parse_args()

    expected = [f"{i}.png" for i in range(FIRST_ID, LAST_ID + 1)]
    problems = []

    for name in expected:
        path = args.denoised_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        with Image.open(path) as image:
            if image.format != "PNG":
                problems.append(f"{name}: format is {image.format}, expected PNG")
            if image.mode != "RGB":
                problems.append(f"{name}: mode is {image.mode}, expected RGB")
            size = image.size
        noisy = args.noise_dir / f"{path.stem}_noise.png"
        if noisy.is_file():
            with Image.open(noisy) as reference:
                if reference.size != size:
                    problems.append(f"{name}: {size} does not match input {reference.size}")

    extra = sorted(p.name for p in args.denoised_dir.glob("*")
                   if p.is_file() and p.name not in set(expected))
    if extra:
        problems.append(f"unexpected files in {args.denoised_dir}: {', '.join(extra)}")

    if problems:
        print("Submission validation FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in expected:
            archive.write(args.denoised_dir / name, arcname=name)

    size_mb = args.out.stat().st_size / (1 << 20)
    print(f"[ok] {len(expected)} images validated (PNG, RGB, matching dimensions)")
    print(f"[ok] {args.out} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
