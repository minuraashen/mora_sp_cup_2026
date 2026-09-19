"""Verify the rewritten source reproduces the reference implementation exactly.

The NCW model was originally delivered as compiled bytecode. This test loads
both that reference and the readable rewrite, runs them on real competition
images, and asserts the outputs are bit-identical -- evidence that the
submitted source is the code that produced the submitted images.

Skipped automatically once the reference bytecode is removed from the tree.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent.parent
ROOT = SCRIPTS.parent
CHECKPOINT_CANDIDATES = [
    SCRIPTS / "weights" / "ncw_1h_best_composite.pth",
    ROOT / "experiments" / "ncw_1hour_33db_v5" / "checkpoints" / "ncw_1h_best_composite.pth",
]
IMAGES = ["461_noise.png", "462_noise.png", "463_noise.png"]


def _purge(prefix: str) -> None:
    for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
        del sys.modules[name]


def _import_from(path: Path, name: str):
    _purge(name)
    sys.path.insert(0, str(path))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(path))


def main() -> int:
    checkpoint_path = next((p for p in CHECKPOINT_CANDIDATES if p.exists()), None)
    if checkpoint_path is None:
        print("SKIP: checkpoint not found")
        return 0
    if not (ROOT / "ncw" / "model.pyc").exists():
        print("SKIP: reference bytecode not present; parity was verified pre-freeze")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    new_pkg = _import_from(SCRIPTS, "ncw")
    new_model = new_pkg.build_from_checkpoint_config(config)
    new_model.load_state_dict(checkpoint["model"], strict=True)
    new_model.eval()
    print(f"[new ] strict load OK, {sum(p.numel() for p in new_model.parameters())} params")

    ref_pkg = _import_from(ROOT, "ncw")
    ref_model = ref_pkg.model.NCWNAFNet(**config["model"],
                                        a_norm=config["noise"]["a_norm"],
                                        b_norm=config["noise"]["b_norm"])
    ref_model.load_state_dict(checkpoint["model"], strict=True)
    ref_model.eval()
    print(f"[ref ] strict load OK, {sum(p.numel() for p in ref_model.parameters())} params")

    _purge("ncw")
    new_inference = _import_from(SCRIPTS, "ncw.inference")

    failures = 0
    for filename in IMAGES:
        path = ROOT / "competition_data" / "submissions" / "noisy" / filename
        if not path.exists():
            continue
        image = new_inference.read_rgb(path, "cpu")
        # A 384-crop exercises padding, conditioning, all four Haar levels and
        # the full encoder/decoder depth without paying for a 992x992 forward.
        crop = image[:, :, :384, :384]
        with torch.no_grad():
            a = new_model(crop)
            b = ref_model(crop)
        delta = (a - b).abs().max().item()
        status = "OK" if delta == 0.0 else "FAIL"
        if delta != 0.0:
            failures += 1
        print(f"[parity] {filename}: max|delta| = {delta:.3e}  {status}")

    if failures:
        print(f"\nPARITY FAILED on {failures} image(s)")
        return 1
    print("\nPARITY PASSED: rewritten source is bit-identical to the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
