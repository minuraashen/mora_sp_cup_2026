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

The brief also requires the exact commit SHA to be stated in the team README. A commit
cannot contain its own hash -- that is a hash fixed point -- so the stated SHA is
necessarily the last *content* commit, and HEAD is the commit that wrote it. That is
only acceptable if the offset is provably inert, so this tool enforces it:

    --stamp   writes HEAD's SHA into the FROZEN-COMMIT block of scripts/README.md.
    (default) verifies the stated SHA is HEAD itself or HEAD's parent, and in the
              parent case that the ONLY difference between them is that block.

Anything else -- a stale SHA, a code change smuggled into the stamping commit, an
unrelated parent -- exits non-zero.

    python scripts/tools/freeze.py --stamp   # then: git commit -a
    python scripts/tools/freeze.py           # verify + print the record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent

EXPECTED_IDS = [f"{i:03d}" for i in range(461, 481)]

README = SCRIPTS / "README.md"
BEGIN = "<!-- FROZEN-COMMIT-BEGIN -->"
END = "<!-- FROZEN-COMMIT-END -->"
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")


def stated_sha(text: str) -> str | None:
    """The SHA inside the README's FROZEN-COMMIT block, if any."""
    try:
        block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    except IndexError:
        return None
    m = SHA_RE.search(block)
    return m.group(0) if m else None


def stamp(head: str) -> None:
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"ERROR: {README} has no FROZEN-COMMIT block to stamp.")
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    README.write_text(f"{pre}{BEGIN}\n`{head}`\n{END}{post}", encoding="utf-8")


def verify_offset(stated: str, head: str) -> tuple[bool, str]:
    """The stated SHA must be HEAD, or HEAD's parent differing only by the block."""
    if stated == head:
        return True, "stated SHA is HEAD"
    parent = git("rev-parse", "HEAD~1")
    if stated != parent:
        return False, (f"stated {stated[:12]} is neither HEAD ({head[:12]}) nor its "
                       f"parent ({parent[:12] or '?'}) -- re-stamp and commit")
    changed = [f for f in git("diff", "--name-only", stated, head).splitlines() if f]
    if changed != ["scripts/README.md"]:
        return False, f"the stamping commit also changed: {changed[:5]}"
    # ...and within that file, only the block.
    body = git("diff", "-U0", stated, head, "--", "scripts/README.md")
    edits = [ln for ln in body.splitlines()
             if (ln.startswith("+") or ln.startswith("-"))
             and not ln.startswith(("+++", "---"))]
    if not all(SHA_RE.search(ln) for ln in edits):
        return False, f"the stamping commit changed more than the SHA line: {edits[:4]}"
    return True, f"HEAD is a README-only child of the stated commit ({len(edits)} line(s))"


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
    ap.add_argument("--stamp", action="store_true",
                    help="write HEAD's SHA into the README's FROZEN-COMMIT block, "
                         "then commit that one-line change")
    args = ap.parse_args()

    if args.stamp:
        head = git("rev-parse", "HEAD")
        if not head:
            print("ERROR: not a git repository, or git is unavailable.")
            return 1
        stamp(head)
        print(f"stamped scripts/README.md with {head}")
        print("\nNow make that the final commit:\n"
              "    git commit -a -m \"Record the frozen submission commit SHA\"\n"
              "    python scripts/tools/freeze.py")
        return 0

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
    stated = stated_sha(README.read_text(encoding="utf-8")) if README.exists() else None
    ok_offset, why = verify_offset(stated, head) if (stated and head) else (False, "no SHA stamped")

    print(f"Team:              Voxel")
    print(f"Branch:            {branch or '(git unavailable)'}")
    print(f"Git commit SHA:    {head or '(git unavailable)'}   <- HEAD, what the organizers record")
    print(f"README states:     {stated or '(unstamped)'}   <- the frozen content commit")
    print(f"Offset check:      {'OK' if ok_offset else 'FAILED'} -- {why}")
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
    if not ok_offset:
        print("\nERROR: the README's frozen-commit SHA is not consistent with HEAD.\n"
              "       Run `python scripts/tools/freeze.py --stamp`, then commit.")
        return 1
    print("\nWorking tree is clean and the frozen-commit SHA is consistent with HEAD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
