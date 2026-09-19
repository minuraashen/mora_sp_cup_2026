"""
check_report.py -- structural validation of the LaTeX report.

The report is 40% of the grade and a compile error in it is an expensive, entirely
avoidable failure. If a LaTeX engine is available this compiles the document and
reports the page count (the limit is 5). Otherwise it falls back to structural checks
that catch the realistic failure modes: unbalanced environments or braces, a missing
figure or \\input target, and an undefined custom macro.

    python scripts/tools/check_report.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
REPORT = SCRIPTS / "report"

PAGE_LIMIT = 5
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def strip_comments(src: str) -> str:
    out = []
    for line in src.splitlines():
        i, esc = 0, False
        cut = len(line)
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                cut = i
                break
            else:
                esc = False
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=REPORT / "main.tex")
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    print(f"CHECKING {args.tex}\n")
    raw = args.tex.read_text(encoding="utf-8")
    src = strip_comments(raw)

    # environments balance
    begins = re.findall(r"\\begin\{([^}]+)\}", src)
    ends = re.findall(r"\\end\{([^}]+)\}", src)
    from collections import Counter
    diff = Counter(begins) - Counter(ends)
    diff2 = Counter(ends) - Counter(begins)
    check("all environments balanced", not diff and not diff2,
          f"unclosed={dict(diff)} unopened={dict(diff2)}")

    # brace balance
    depth = 0
    esc = False
    bad = False
    for ch in src:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                bad = True
                break
    check("braces balanced", depth == 0 and not bad, f"final depth {depth}")

    # referenced files exist
    missing = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        if not (REPORT / m.group(1)).exists():
            missing.append(m.group(1))
    for m in re.finditer(r"\\input\{([^}]+)\}", src):
        t = m.group(1)
        if not (REPORT / t).exists() and not (REPORT / (t + ".tex")).exists():
            missing.append(t)
    check("all \\includegraphics / \\input targets exist", not missing, str(missing))

    # Custom macros. Anything matching our CamelCase naming convention that is used but
    # never defined would compile to an "Undefined control sequence" error.
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", src))
    body = src.split(r"\begin{document}", 1)[-1]
    used_caps = set(re.findall(r"\\([A-Z][A-Za-z]{2,})(?![A-Za-z])", body))
    latex_builtin = {"LaTeX", "TeX", "Large", "Delta", "Longleftrightarrow", "Leftrightarrow"}
    undefined = sorted(used_caps - defined - latex_builtin)
    check("no undefined CamelCase macros in the body", not undefined, str(undefined))

    unused = sorted(defined - used_caps)
    check("every \\newcommand is used", not unused, str(unused))

    # placeholders left in
    tbd = sorted(set(re.findall(r"\bTBD\b", src)))
    check("no TBD placeholders left", not tbd, f"{len(re.findall(r'TBD', src))} occurrence(s)")

    # formatting requirements
    check("12pt documentclass", "12pt" in src)
    check("1in margins", re.search(r"margin=1in", src) is not None)
    check("no titlepage / \\maketitle", "\\maketitle" not in src and "titlepage" not in src)

    # compile if we can
    engine = None
    for cand in ("tectonic", "pdflatex", "latexmk"):
        if shutil.which(cand):
            engine = cand
            break
    if args.no_compile or not engine:
        print(f"\n  [skip] no LaTeX engine found -- compile on Overleaf and confirm "
              f"the page count is <= {PAGE_LIMIT}")
    else:
        print(f"\ncompiling with {engine} ...")
        with tempfile.TemporaryDirectory() as td:
            if engine == "tectonic":
                cmd = [engine, "-o", td, "--keep-logs", str(args.tex)]
            else:
                cmd = [engine, "-interaction=nonstopmode", f"-output-directory={td}", str(args.tex)]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPORT))
            pdf = next(Path(td).glob("*.pdf"), None)
            if not check("document compiles", pdf is not None):
                print((r.stdout + r.stderr)[-2500:])
            else:
                try:
                    from pypdf import PdfReader
                    n = len(PdfReader(str(pdf)).pages)
                except Exception:
                    n = raw_pages(pdf)
                check(f"page count <= {PAGE_LIMIT}", n is None or n <= PAGE_LIMIT,
                      f"{n} pages")
                shutil.copy2(pdf, REPORT / "Voxel_Report.pdf")
                print(f"  wrote {REPORT / 'Voxel_Report.pdf'}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        return 1
    print("REPORT CHECKS PASSED")
    return 0


def raw_pages(pdf: Path):
    try:
        data = pdf.read_bytes()
        n = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
        return n if n > 0 else None
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
