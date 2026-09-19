"""
preflight.py -- the submission validator.

Every check here corresponds to a stated disqualification or penalty condition. It is
meant to be run immediately before submitting and again after any change.

The two that actually matter most:
  * REPRODUCTION -- re-runs denoise.py in a clean subprocess into a fresh directory and
    compares SHA-256 against the files being submitted. "The submitted code must
    reproduce the submitted images" is a DQ rule, and this is the only real evidence.
  * RENAME TEST -- copies 461_noise.png to 999_noise.png, runs the pipeline, and
    asserts the output is byte-identical to 461.png. If any per-image lookup or
    hardcoded id existed, this would differ. It is the strongest available proof that
    the pipeline is a single generalisable procedure.

    python scripts/tools/preflight.py --pred_dir competition_data/submissions/denoised
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

EXPECTED_IDS = [f"{i:03d}" for i in range(461, 481)]
NETWORK_TOKENS = ("urllib", "requests", "torch.hub", "http://", "https://",
                  "pretrained=True", "download_url", "hf_hub")
RUNTIME_BUDGET_S = 60.0

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path,
                    default=REPO / "competition_data" / "submissions" / "denoised")
    ap.add_argument("--noisy_dir", type=Path,
                    default=REPO / "competition_data" / "submissions" / "noisy")
    ap.add_argument("--zip", type=Path, default=REPO / "Voxel.zip")
    ap.add_argument("--readme", type=Path, default=SCRIPTS / "README.md")
    ap.add_argument("--weights", type=Path, default=SCRIPTS / "weights" / "voxel_nafnet.pth")
    ap.add_argument("--skip_repro", action="store_true",
                    help="skip the (slow) reproduction and rename checks")
    args = ap.parse_args()

    print("PREFLIGHT\n")

    # 1 -- the entry point runs
    r = subprocess.run([sys.executable, str(SCRIPTS / "denoise.py"), "--help"],
                       capture_output=True, text=True)
    check("denoise.py --help exits 0", r.returncode == 0, r.stderr.strip()[:120])

    # 2 -- file set
    files = sorted(args.pred_dir.glob("*.png")) if args.pred_dir.is_dir() else []
    stems = [p.stem for p in files]
    check("exactly 20 PNGs named 461..480", stems == EXPECTED_IDS,
          f"found {len(stems)}: {stems[:3]}{'...' if len(stems) > 3 else ''}")
    extras = [p.name for p in args.pred_dir.iterdir()
              if p.is_dir() or p.suffix.lower() != ".png"] if args.pred_dir.is_dir() else []
    check("no extra files or subdirectories in the output folder", not extras, str(extras[:5]))

    # 3 -- every image is a valid RGB 992x992 PNG, and actually changed
    bad, unchanged, bright, signed = [], [], [], []
    for p in files:
        with Image.open(p) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != (992, 992):
                bad.append(f"{p.name}:{im.format}/{im.mode}/{im.size}")
                continue
            a = np.asarray(im, dtype=np.uint8)
        if a.max() == 0:
            bad.append(f"{p.name}:all-zero")
        src = args.noisy_dir / f"{p.stem}_noise.png"
        if src.exists():
            n = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
            if np.array_equal(a, n):
                unchanged.append(p.name)
            # 5 -- no deliberate brightening.
            # A correct denoiser DOES move the mean: on the public pairs the clean
            # image differs from its noisy version by -5.9 to +3.8 levels, so a
            # per-image shift of a few levels is required, not suspicious. What a
            # gamma/Retinex stage would produce instead is a SYSTEMATIC positive shift
            # on every image (gamma 1.05 is roughly +4.6 levels everywhere), so the
            # discriminating statistic is the mean over the whole set.
            d = (float(a.mean()) - float(n.mean())) / 255.0
            signed.append(d)
            if abs(d) > 8.0 / 255.0:
                bright.append(f"{p.name}:{d * 255:+.2f}")
    check("all PNG / RGB / 992x992 / non-degenerate", not bad, str(bad[:3]))
    check("output differs from input", not unchanged, str(unchanged[:3]))
    check("no extreme per-image brightness shift (>8/255)", not bright, str(bright[:3]))
    if signed:
        sys_shift = float(np.mean(signed)) * 255.0
        check("no systematic brightening (|mean shift| <= 1 level)", abs(sys_shift) <= 1.0,
              f"mean shift {sys_shift:+.2f} levels over {len(signed)} images")

    # 7 -- offline: nothing in the graded path may touch the network
    hits = []
    for py in sorted(SCRIPTS.glob("*.py")) + sorted((SCRIPTS / "spcup").glob("*.py")):
        src = py.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                names = [a.name for a in getattr(node, "names", [])]
                for t in ("urllib", "requests", "socket", "http"):
                    if mod.startswith(t) or any(n.startswith(t) for n in names):
                        hits.append(f"{py.name}:{t}")
        for tok in ("torch.hub", "pretrained=True", "hf_hub"):
            if tok in src:
                hits.append(f"{py.name}:{tok}")
    check("graded path has no network access", not hits, str(sorted(set(hits))[:5]))

    # 8 -- model freeze
    if args.weights.exists():
        digest = sha256(args.weights)
        readme = args.readme.read_text(encoding="utf-8", errors="ignore") if args.readme.exists() else ""
        check("checkpoint SHA-256 recorded in scripts/README.md", digest in readme,
              f"sha256={digest[:16]}...")
        check("README has no unfilled placeholders",
              "<commit SHA>" not in readme and "<SHA-256" not in readme)
    else:
        check("checkpoint present", False, str(args.weights))

    # 10 -- scope: only scripts/ (and our own docs/analysis) may change
    r = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
    touched = [ln[3:].strip().strip('"') for ln in r.stdout.splitlines() if ln.strip()]
    forbidden = [t for t in touched
                 if t.startswith(("baseline/", "evaluation/", "competition_data/"))]
    check("no modifications under baseline/ evaluation/ competition_data/",
          not forbidden, str(forbidden[:5]))

    # 11a -- no hardcoded ids.
    # Searched against the CODE only: docstrings and comments legitimately mention
    # example filenames such as "461_noise.png -> 461.png", and flagging those would
    # be a false positive that punishes documenting the code.
    def strip_docstrings(tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
        return tree

    id_hits = []
    for py in sorted(SCRIPTS.glob("*.py")) + sorted((SCRIPTS / "spcup").glob("*.py")):
        try:
            code = ast.unparse(strip_docstrings(ast.parse(
                py.read_text(encoding="utf-8", errors="ignore"))))
        except (SyntaxError, AttributeError):
            code = py.read_text(encoding="utf-8", errors="ignore")
        for i in range(461, 501):
            if str(i) in code:
                id_hits.append(f"{py.name}:{i}")
    check("no submission image ids appear in the graded code", not id_hits, str(id_hits[:5]))

    # 6 + 11b -- reproduction and the rename test
    if not args.skip_repro and files:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src_in, out_a = td / "in", td / "a"
            src_in.mkdir()
            for i in (461, 462):
                s = args.noisy_dir / f"{i:03d}_noise.png"
                if s.exists():
                    shutil.copy2(s, src_in / s.name)
            # the rename test: same content, a different id
            s461 = args.noisy_dir / "461_noise.png"
            if s461.exists():
                shutil.copy2(s461, src_in / "999_noise.png")

            r = subprocess.run([sys.executable, str(SCRIPTS / "denoise.py"),
                                "--noise_dir", str(src_in), "--denoised_dir", str(out_a), "--quiet"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                check("reproduction subprocess ran", False, r.stderr.strip()[-200:])
            else:
                same = []
                for i in (461, 462):
                    a, b = out_a / f"{i:03d}.png", args.pred_dir / f"{i:03d}.png"
                    if a.exists() and b.exists():
                        same.append(sha256(a) == sha256(b))
                check("re-run reproduces the submitted PNGs byte-for-byte",
                      bool(same) and all(same), f"checked {len(same)} image(s)")

                a999, a461 = out_a / "999.png", out_a / "461.png"
                if a999.exists() and a461.exists():
                    check("rename test: 999 output identical to 461 (no per-image logic)",
                          sha256(a999) == sha256(a461))
                else:
                    check("rename test produced outputs", False)

    # 12 -- the zip
    if args.zip.exists():
        with zipfile.ZipFile(args.zip) as z:
            names = sorted(z.namelist())
        check("zip contains exactly the 20 flat PNGs",
              names == [f"{i}.png" for i in EXPECTED_IDS], str(names[:3]))
        check("zip has no __MACOSX/.DS_Store",
              not any(("__MACOSX" in n or ".DS_Store" in n) for n in names))

        # The zip is what the organizers score, so its CONTENTS -- not just its member
        # names -- must be exactly the files the pipeline produced. A stale zip built
        # before the final model would pass every other check here.
        mismatched = []
        with zipfile.ZipFile(args.zip) as z:
            for i in EXPECTED_IDS:
                local = args.pred_dir / f"{i}.png"
                if not local.exists():
                    mismatched.append(f"{i}:missing-local")
                    continue
                if hashlib.sha256(z.read(f"{i}.png")).hexdigest() != sha256(local):
                    mismatched.append(i)
        check("zip contents byte-identical to the rendered PNGs",
              not mismatched, str(mismatched[:5]))
    else:
        print(f"  [skip] zip not built yet ({args.zip})")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    if n_fail:
        print(f"{n_fail} CHECK(S) FAILED")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
