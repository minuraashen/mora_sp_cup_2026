"""
freeze.py -- produce the submission freeze record.

The handbook freezes three things independently, and a mismatch between any two of
them is a penalty or a disqualification:

  * the CODE, identified by the git commit SHA stated on the submission form;
  * the MODEL, identified by the SHA-256 of the checkpoint;
  * the IMAGES, which the submitted code must reproduce exactly.

`preflight.py` proves the third by re-running the pipeline. What was missing is the
durable record of *what* was frozen: a manifest that pins the SHA-256 of every
submitted PNG, of `Voxel.zip`, and of the checkpoint, so that after the deadline the
submitted bundle can be re-identified byte for byte without re-running anything.

This writes `scripts/artifacts/submission_manifest.json` and prints the block to paste
into the organizers' submission form.

The commit SHA is deliberately NOT written into the manifest: a commit cannot contain
its own hash, so recording one here would necessarily name the *previous* commit and
would be wrong the moment this file is committed. It is printed live instead, read
from `git rev-parse HEAD`, which is the value the organizers record.

    python scripts/tools/freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent

EXPECTED_IDS = [f"{i:03d}" for i in range(461, 481)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path,
                    default=REPO / "competition_data" / "submissions" / "denoised")
    ap.add_argument("--zip", type=Path, default=REPO / "Voxel.zip")
    ap.add_argument("--weights", type=Path, default=SCRIPTS / "weights" / "voxel_nafnet.pth")
    ap.add_argument("--out", type=Path,
                    default=SCRIPTS / "artifacts" / "submission_manifest.json")
    args = ap.parse_args()

    missing = [i for i in EXPECTED_IDS if not (args.pred_dir / f"{i}.png").exists()]
    if missing:
        print(f"ERROR: missing predictions: {missing}")
        print("Run scripts/denoise.py first.")
        return 1
    if not args.weights.exists():
        print(f"ERROR: checkpoint not found: {args.weights}")
        return 1

    images = {i: sha256(args.pred_dir / f"{i}.png") for i in EXPECTED_IDS}
    model_sha = sha256(args.weights)
    zip_sha = sha256(args.zip) if args.zip.exists() else None

    manifest = {
        "team": "Voxel",
        "round": "preliminary",
        "entry_point": "scripts/denoise.py",
        "command": ("python scripts/denoise.py "
                    "--noise_dir competition_data/submissions/noisy "
                    "--denoised_dir competition_data/submissions/denoised"),
        "model": {
            "path": "scripts/weights/voxel_nafnet.pth",
            "bytes": args.weights.stat().st_size,
            "sha256": model_sha,
        },
        "zip": None if zip_sha is None else {
            "name": args.zip.name,
            "bytes": args.zip.stat().st_size,
            "sha256": zip_sha,
        },
        "images": images,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO)}")

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")

    print("\n" + "=" * 70)
    print("SUBMISSION FREEZE RECORD -- paste into the organizers' form")
    print("=" * 70)
    print(f"Team:              Voxel")
    print(f"Branch:            {branch or '(git unavailable)'}")
    print(f"Git commit SHA:    {head or '(git unavailable)'}")
    print(f"Model file:        voxel_nafnet.pth")
    print(f"Model local path:  scripts/weights/voxel_nafnet.pth  (committed; no download needed)")
    print(f"Model SHA-256:     {model_sha}")
    if zip_sha:
        print(f"Voxel.zip SHA-256: {zip_sha}")
    else:
        print(f"Voxel.zip:         NOT BUILT -- run scripts/tools/make_submission.py")
    print("=" * 70)

    if dirty:
        n = len(dirty.splitlines())
        print(f"\nWARNING: {n} uncommitted change(s) in the working tree. The commit SHA "
              f"above\n         does not describe what is on disk. Commit, then re-run "
              f"this tool\n         and report the NEW SHA.")
        return 1
    if zip_sha is None:
        return 1
    print("\nWorking tree is clean; the SHA above is the submittable commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
