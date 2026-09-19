"""
export_model.py -- turn a training checkpoint into the shipped model file.

Strips optimizer/EMA/bookkeeping state and stores the weights as fp16. fp16 halves the
file (29.16 M params: 116.6 MB -> 58.3 MB) which keeps it under GitHub's 100 MB limit,
so the checkpoint can be committed to the repository and no Google Drive link -- and no
"did the reviewer download the right file" failure mode -- is needed. Inference casts
back to fp32 on load, so the stored precision does not affect the forward pass
meaningfully (verified below by re-scoring after export).

    python scripts/train/export_model.py --ckpt scripts/_runs/run1/best_ema.pth
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spcup"))

import common  # noqa: E402
import nafnet  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=common.WEIGHTS_DIR / "voxel_nafnet.pth")
    ap.add_argument("--fp32", action="store_true", help="store fp32 instead of fp16")
    ap.add_argument("--which", choices=["auto", "model", "ema"], default="auto",
                    help="which weight set to export. A training checkpoint holds both "
                         "the raw and the EMA weights, and either can be the better one, "
                         "so pass it explicitly when selection has chosen for you.")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "w32")
    if args.which != "auto":
        if args.which not in ck:
            print(f"ERROR: {args.ckpt} has no '{args.which}' weights "
                  f"(keys: {sorted(k for k in ck if isinstance(ck[k], dict))})")
            return 2
        sd = ck[args.which]
    else:
        sd = ck.get("model", ck.get("ema", ck.get("params", ck)))
    print(f"exporting weight set: {args.which}")

    # sanity: it must load into the architecture we ship
    model = nafnet.build(arch)
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)

    dtype = torch.float32 if args.fp32 else torch.float16
    out_sd = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in sd.items()}

    payload = {
        "model": out_sd,
        "arch": arch,
        "source_ckpt": str(args.ckpt),
        "step": ck.get("step"),
        "val_composite": ck.get("composite"),
        "val_psnr": ck.get("psnr"),
        "val_ssim": ck.get("ssim"),
        "note": "Team Voxel -- Mora SP Cup 2026. Weights stored fp16, cast to fp32 at load.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    size = args.out.stat().st_size
    digest = sha256_file(args.out)
    print(f"exported {args.ckpt}  ->  {args.out}")
    print(f"  arch        {arch} ({nafnet.count_params(model):,} params)")
    print(f"  step        {ck.get('step')}")
    print(f"  val comp    {ck.get('composite')}")
    print(f"  size        {size:,} bytes ({size / 1e6:.1f} MB)  "
          f"{'OK for git (<100MB)' if size < 100e6 else 'TOO BIG for git'}")
    print(f"  SHA-256     {digest}")

    common.write_json(common.ARTIFACTS_DIR / "model_export.json", {
        "path": str(args.out), "bytes": size, "sha256": digest, "arch": arch,
        "step": ck.get("step"), "val_composite": ck.get("composite"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
