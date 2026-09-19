"""
train.py -- fine-tune NAFNet on the 460 public pairs (+ measured synthetic noise).

Selection is on the OFFICIAL validation composite, never on loss. Checkpoints are
written continuously so there is always a shippable model on disk.

    python scripts/train/train.py --run run1 --init scripts/weights/NAFNet-SIDD-width32.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spcup"))

import common  # noqa: E402
import nafnet  # noqa: E402
import dataset as ds_mod  # noqa: E402
import losses as loss_mod  # noqa: E402
import noise_model  # noqa: E402


def parse_patch_schedule(spec: str):
    """'0:256x4,36000:320x3' -> [(0,256,4), (36000,320,3)]"""
    out = []
    for part in spec.split(","):
        at, cfg = part.split(":")
        p, b = cfg.lower().split("x")
        out.append((int(at), int(p), int(b)))
    return sorted(out)


def lr_at(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * min(1.0, t)))


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


@torch.no_grad()
def validate(model, ids, noisy_mm, clean_mm, baseline, device, tag=""):
    """Full-resolution validation on the official composite."""
    model.eval()
    comps, psnrs, ssims = [], [], []
    for i in ids:
        k = i - 1
        n = torch.from_numpy(np.ascontiguousarray(noisy_mm[k])).permute(2, 0, 1)[None]
        g = torch.from_numpy(np.ascontiguousarray(clean_mm[k])).permute(2, 0, 1)[None]
        n = n.to(device).float().div_(255.0)
        g = g.to(device).float().div_(255.0)
        out = model(n).clamp_(0, 1)
        bp = torch.tensor([baseline[str(i)]["psnr"]], device=device, dtype=torch.float64)
        bs = torch.tensor([baseline[str(i)]["ssim"]], device=device, dtype=torch.float64)
        c, p, s = common.composite_from_tensors(out, g, bp, bs)
        comps.append(float(c[0])); psnrs.append(float(p[0])); ssims.append(float(s[0]))
    model.train()
    return float(np.mean(comps)), float(np.mean(psnrs)), float(np.mean(ssims))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    ap.add_argument("--arch", default="w32", choices=sorted(nafnet.ARCHS))
    ap.add_argument("--init", default="", help="checkpoint to start from, or '' / 'none'")
    ap.add_argument("--iters", type=int, default=64000)
    ap.add_argument("--patch_schedule", default="0:256x4")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ema_start", type=int, default=2000)
    ap.add_argument("--p_synth", type=float, default=0.4)
    ap.add_argument("--ssim_w", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_every", type=int, default=2500)
    ap.add_argument("--fullval_every", type=int, default=10000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=0.0, help="wall-clock budget; 0 = none")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.iters = min(args.iters, 200)
        args.val_every = 100
        args.fullval_every = 10 ** 9
        args.ema_start = 50

    common.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    run_dir = common.RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,lr,loss,val_composite,val_psnr,val_ssim,ema,secs\n", encoding="utf-8")

    # ---- data
    split = common.load_split()
    baseline = common.read_json(common.ARTIFACTS_DIR / "noisy_baseline.json")
    noisy_mm = np.load(common.ARTIFACTS_DIR / "noisy_u8.npy", mmap_mode="r")
    clean_mm = np.load(common.ARTIFACTS_DIR / "clean_u8.npy", mmap_mode="r")

    hot = np.load(common.ARTIFACTS_DIR / "hotpixels.npy")
    S = common.IMG_SIZE
    hot_mask = torch.zeros((S, S), dtype=torch.bool, device=device)
    hot_mask[torch.from_numpy(hot[:, 0].astype(np.int64)),
             torch.from_numpy(hot[:, 1].astype(np.int64))] = True

    schedule = parse_patch_schedule(args.patch_schedule)
    patch, batch = schedule[0][1], schedule[0][2]

    train_ds = ds_mod.PairedPatches(split["train"], patch=patch, p_synth=args.p_synth,
                                    seed=args.seed, length=10 ** 7)

    # ---- model
    model = nafnet.build(args.arch).to(device)
    if args.init and args.init.lower() != "none":
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        sd = ck.get("params", ck.get("model", ck))
        model.load_state_dict(sd, strict=True)
        print(f"[init] loaded {args.init} (strict)")
    else:
        print("[init] training from scratch")
    print(f"[model] arch={args.arch} params={nafnet.count_params(model):,}")

    ema = EMA(model, args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    criterion = loss_mod.CombinedLoss(args.ssim_w).to(device)
    synth = noise_model.NoiseSynthesizer(device, torch.Generator(device=device).manual_seed(args.seed))

    best = {}   # keyed "raw_46" / "ema_46" / "raw_23" / "ema_23"
    step = 0
    t_start = time.time()
    last_hour_snap = -1
    loader = None

    def make_loader(p, b):
        train_ds.set_patch(p)
        return DataLoader(train_ds, batch_size=b, shuffle=False, num_workers=args.num_workers,
                          pin_memory=(device == "cuda"), drop_last=True)

    loader = make_loader(patch, batch)
    it = iter(loader)
    model.train()
    loss_acc, loss_n = 0.0, 0

    while step < args.iters:
        # patch-size schedule
        for at, p, b in schedule:
            if step == at and (p != patch or b != batch):
                patch, batch = p, b
                loader = make_loader(patch, batch)
                it = iter(loader)
                print(f"[sched] step {step}: patch={patch} batch={batch}", flush=True)

        try:
            noisy_u8, clean_u8, meta, is_synth = next(it)
        except StopIteration:
            train_ds.epoch += 1
            it = iter(loader)
            continue

        clean = clean_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        noisy = noisy_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        is_synth = is_synth.to(device)

        if bool(is_synth.any()):
            m = ds_mod.transform_mask(hot_mask, meta, patch)
            fake = synth(clean, m)
            sel = is_synth.view(-1, 1, 1, 1)
            noisy = torch.where(sel, fake, noisy)

        lr = lr_at(step, args.iters, args.warmup, args.lr, args.lr_min)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            pred = model(noisy)
            loss = criterion(pred, clean)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()

        if step >= args.ema_start:
            ema.update(model)

        loss_acc += float(loss.detach()); loss_n += 1
        step += 1

        # guard against the silent VRAM-spill failure mode
        if step == 20 and device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[vram] peak after 20 steps: {peak:.2f} GB", flush=True)
            if peak > 5.0:
                print("ABORT: peak VRAM > 5.0 GB, this config will spill to system memory.")
                return 1

        if step % 100 == 0:
            el = time.time() - t_start
            print(f"[{step:6d}/{args.iters}] loss={loss_acc / max(1, loss_n):8.4f} "
                  f"lr={lr:.2e} {el / step:.3f}s/it elapsed={el / 60:.1f}m", flush=True)
            loss_acc, loss_n = 0.0, 0

        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                        "step": step, "arch": args.arch, "args": vars(args)},
                       run_dir / "last.pth")

        hour = int((time.time() - t_start) // 3600)
        if hour > last_hour_snap:
            last_hour_snap = hour
            torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                        "step": step, "arch": args.arch}, run_dir / f"hour_{hour:02d}.pth")

        if step % args.val_every == 0 or step == args.iters:
            # The final step always gets a FULL validation: it is the fully annealed
            # model and the most likely winner, so it must not be scored on the subset.
            use_full = (step % args.fullval_every == 0) or (step == args.iters)
            ids = split["val"] if use_full else split["fast_val"]
            for which, net in (("raw", model), ("ema", ema.shadow)):
                c, p, s = validate(net, ids, noisy_mm, clean_mm, baseline, device)
                # Best-tracking is keyed by subset size: a 23-image score and a
                # 46-image score are not comparable (the full set runs ~0.03 lower),
                # so comparing them across subsets would pick checkpoints by which
                # subset they happened to be evaluated on.
                which = f"{which}_{len(ids)}"
                el = time.time() - t_start
                print(f"    VAL[{which}] n={len(ids)} composite={c:.4f} psnr={p:.3f} ssim={s:.4f}",
                      flush=True)
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"{step},{lr:.6e},{loss:.6f},{c:.6f},{p:.4f},{s:.6f},{which},{el:.0f}\n")
                if c > best.get(which, -1.0):
                    best[which] = c
                    torch.save({"model": net.state_dict(), "step": step, "arch": args.arch,
                                "composite": c, "psnr": p, "ssim": s, "args": vars(args)},
                               run_dir / f"best_{which}.pth")
                    print(f"    -> new best_{which} ({c:.4f})", flush=True)

        if args.max_minutes and (time.time() - t_start) / 60.0 > args.max_minutes:
            print(f"[stop] wall-clock budget {args.max_minutes} min reached at step {step}")
            break

    common.write_json(run_dir / "summary.json", {
        "run": args.run, "arch": args.arch, "steps": step,
        "best": best,
        "minutes": (time.time() - t_start) / 60.0, "args": vars(args),
    })
    print("\ndone: " + "  ".join(f"best_{k}={v:.4f}" for k, v in sorted(best.items()))
          + f"  in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
