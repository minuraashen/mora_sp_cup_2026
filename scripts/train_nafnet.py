"""Fine-tune NAFNet on a fixed training split; select EMA by held-out composite."""

from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch

from nafnet.augmentation import synthesize, transform
from nafnet.classical import repair_impulses
from nafnet.inference import load_model, predict, sha256_file
from nafnet.losses import restoration_loss
from nafnet.scoring import score_arrays, summarize


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path}")
        return np.array(image, dtype=np.float32) / 255


def tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)


class PairedCrops:
    """Bounded image cache; training never opens a validation target."""

    def __init__(self, data: Path, ids: list[str], fit: dict, seed: int, cache_size: int = 6,
                 synthetic_probability: float = 0.5):
        self.data, self.ids, self.fit = data, ids, fit
        self.rng = np.random.default_rng(seed)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.synthetic_probability = synthetic_probability

    def batch(self, size: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = [], []
        for _ in range(count):
            image_id = str(self.rng.choice(self.ids))
            if image_id not in self.cache:
                self.cache[image_id] = (
                    load_rgb(self.data / "noisy" / f"{image_id}_noise.png"),
                    load_rgb(self.data / "ground_truth" / f"{image_id}.png"),
                )
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(image_id)
            noisy, clean = self.cache[image_id]
            top = int(self.rng.integers(0, clean.shape[0] - size + 1))
            left = int(self.rng.integers(0, clean.shape[1] - size + 1))
            crop = np.s_[top:top + size, left:left + size]
            clean_crop = clean[crop]
            noisy_crop = (synthesize(clean_crop, self.fit, self.rng)
                          if self.rng.random() < self.synthetic_probability else noisy[crop])
            noisy_crop, _ = repair_impulses(noisy_crop)
            symmetry = int(self.rng.integers(0, 8))
            inputs.append(tensor(transform(noisy_crop, symmetry)))
            targets.append(tensor(transform(clean_crop, symmetry)))
        return torch.cat(inputs), torch.cat(targets)


@torch.no_grad()
def validate(model: torch.nn.Module, data: Path, ids: list[str], device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    for index, image_id in enumerate(ids, 1):
        noisy = load_rgb(data / "noisy" / f"{image_id}_noise.png")
        clean = load_rgb(data / "ground_truth" / f"{image_id}.png")
        repaired, _ = repair_impulses(noisy)
        output = predict(model, tensor(repaired), device, tile=256, overlap=32)
        array = output[0].permute(1, 2, 0).numpy()
        row = {"image_id": image_id, **score_arrays(array, clean, noisy)}
        rows.append(row)
        if index % 10 == 0:
            print(f"  Held-out validation {index}/{len(ids)}", flush=True)
    return summarize(rows), rows


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_state(value):
    """Make optimizer/model state portable, including the native DirectML device."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("competition_data/public"))
    parser.add_argument("--run", type=Path, default=Path("scripts/runs/nafnet_docs_20260918"))
    parser.add_argument("--weights", type=Path, default=Path("scripts/weights/NAFNet-SIDD-width32.pth"))
    parser.add_argument("--output", type=Path, default=Path("scripts/weights/NAFNet-Mora-docs-best.pth"))
    parser.add_argument("--device", choices=("cpu", "cuda", "directml"), default="cuda")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--validate-every", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--benchmark", action="store_true", help="Run 5 update steps; save no weights.")
    parser.add_argument("--benchmark-patch", type=int, choices=(128, 256), default=128)
    parser.add_argument("--resume", type=Path, help="Resume a saved training state, including pending validation.")
    parser.add_argument("--patch-size", type=int, choices=(128, 256),
                        help="Fixed crop size for a continuation phase; default grows 128 to 256.")
    parser.add_argument("--synthetic-probability", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, help="Augmentation seed; does not change the fixed data split.")
    parser.add_argument("--target-score", type=float,
                        help="Stop after validation reaches this composite; otherwise complete --steps.")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.validate_every, args.threads) < 1:
        parser.error("Step, batch, validation and thread counts must be positive.")
    if not 0 <= args.synthetic_probability <= 1 or not 0 <= args.ema_decay < 1:
        parser.error("Require synthetic probability in [0,1] and EMA decay in [0,1).")
    if args.target_score is not None and not 0 < args.target_score <= 1:
        parser.error("Target composite must be in (0,1].")
    if args.output.exists() and not args.benchmark and not args.resume:
        parser.error(f"Refusing to replace an existing experiment checkpoint: {args.output}")
    if not args.benchmark and shutil.disk_usage(args.run).free < 1024**3:
        raise OSError("Free at least 1 GiB before training to allow safe checkpoint writes; the full rerun needs additional space.")
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    fit = json.loads((args.run / "noise_model.json").read_text(encoding="utf-8"))
    if set(split["train"]) & set(split["validation"]) or set(fit["fit_ids"]) != set(split["train"]):
        raise ValueError("Training/validation leakage in split or fitted noise model.")
    seed = split["seed"] if args.seed is None else args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == "directml":
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device(args.device)
    model, config = load_model(args.weights, device)
    model.train().requires_grad_(True)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, foreach=False)
    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sampler = PairedCrops(args.data, split["train"], fit, seed,
                         synthetic_probability=args.synthetic_probability)
    steps = 5 if args.benchmark else args.steps
    started = time.perf_counter()
    best_score = -float("inf")
    history_path = args.run / "training_history.jsonl"
    metadata = {"architecture": config, "seed": seed, "steps": args.steps,
                "batch": args.batch, "lr": args.lr, "device": str(device),
                "torch": str(torch.__version__), "python": platform.python_version(),
                "initial_checkpoint_sha256": sha256_file(args.weights),
                "split_sha256": sha256_file(args.run / "split.json"),
                "noise_model_sha256": sha256_file(args.run / "noise_model.json"),
                "synthetic_probability": args.synthetic_probability, "ema_decay": args.ema_decay,
                "patch_size": args.patch_size, "target_score": args.target_score,
                "loss": "Charbonnier + 0.1 * three-scale SSIM loss + 0.05 * Haar L1",
                "precision": "CUDA mixed precision" if use_amp else "float32"}
    running_loss = 0.0
    start_step = 0
    pending_validation = False
    prior_seconds = 0.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("seed", "steps", "batch", "lr", "split_sha256", "noise_model_sha256", "initial_checkpoint_sha256"):
            if saved["config"][key] != metadata[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        for key, default in (("synthetic_probability", 0.5), ("ema_decay", 0.995), ("patch_size", None)):
            if saved["config"].get(key, default) != metadata[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        torch.set_rng_state(saved["torch_rng"])
        start_step = saved["step"]
        pending_validation = saved["pending_validation"]
        best_score = saved["best_score"]
        running_loss = saved["running_loss"]
        prior_seconds = saved["elapsed_seconds"]
        del saved
    if not args.benchmark:
        (args.run / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Training NAFNet | {device} | {steps} steps | batch {args.batch} | 410 train / 50 val", flush=True)

    def save_restart(step: int, pending: bool) -> None:
        atomic_save({"model": cpu_state(model.state_dict()), "ema": cpu_state(ema.state_dict()),
                     "optimizer": cpu_state(optimizer.state_dict()), "scaler": scaler.state_dict(),
                     "sampler_rng": sampler.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                     "step": step, "pending_validation": pending, "best_score": best_score,
                     "running_loss": running_loss, "config": metadata,
                     "elapsed_seconds": prior_seconds + time.perf_counter() - started},
                    args.run / "training_state.pth")

    def score_checkpoint(step: int) -> None:
        nonlocal best_score
        summary, rows = validate(ema, args.data, split["validation"], device)
        record = {"step": step, "loss": running_loss / step,
                  "elapsed_seconds": prior_seconds + time.perf_counter() - started, **summary}
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"Validation | PSNR {summary['psnr']:.4f} | SSIM {summary['ssim']:.6f} | composite {summary['composite_score']:.8f}", flush=True)
        if summary["composite_score"] > best_score:
            if shutil.disk_usage(args.output.parent).free < 350 * 1024**2:
                raise OSError("At least 350 MiB free is required to safely save a new checkpoint.")
            best_score = summary["composite_score"]
            atomic_save({"params_ema": cpu_state(ema.state_dict()),
                         "training": {**metadata, "selected_step": step, "validation": summary}}, args.output)
            (args.run / "best_validation.json").write_text(json.dumps({"step": step, "summary": summary, "images": rows}, indent=2) + "\n", encoding="utf-8")
            print(f"Selected checkpoint at step {step}: {args.output}", flush=True)
        save_restart(step, False)
        model.train()

    if pending_validation:
        print(f"Resuming pending validation at step {start_step}", flush=True)
        score_checkpoint(start_step)
    completed_step = start_step
    end_step = start_step if args.target_score is not None and best_score >= args.target_score else steps
    for step in range(start_step + 1, end_step + 1):
        # Default progressive schedule; fixed-size continuation keeps the requested batch.
        patch = args.benchmark_patch if args.benchmark else (args.patch_size or (128 if step <= int(steps * 0.7) else 256))
        batch = args.batch if args.patch_size or patch == 128 else max(1, args.batch // 2)
        inputs, targets = sampler.batch(patch, batch)
        inputs, targets = inputs.to(device), targets.to(device)
        rate = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / steps)))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext():
            prediction = model(inputs)
            loss = restoration_loss(prediction.float(), targets)
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss at step {step}.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.mul_(args.ema_decay).add_(current.detach(), alpha=1 - args.ema_decay)
        running_loss += loss_value
        completed_step = step
        if step % 25 == 0 or args.benchmark or step == steps:
            print(f"Step {step}/{steps} | patch {patch} | loss {loss_value:.6f} | elapsed {time.perf_counter() - started:.1f}s", flush=True)
        if not args.benchmark and (step % args.validate_every == 0 or step == steps):
            save_restart(step, True)
            score_checkpoint(step)
            if args.target_score is not None and best_score >= args.target_score:
                print(f"Validation target {args.target_score:.6f} reached at step {step}; stopping.", flush=True)
                break
    elapsed = prior_seconds + time.perf_counter() - started
    print(f"{'Benchmark' if args.benchmark else 'Training'} complete | {elapsed:.2f}s", flush=True)
    if not args.benchmark:
        print(f"Best held-out composite: {best_score:.8f} | SHA-256 {sha256_file(args.output)}", flush=True)
        (args.run / "training_finished.json").write_text(json.dumps({
            "planned_steps": args.steps, "completed_steps": completed_step,
            "elapsed_seconds": elapsed, "best_composite": best_score,
            "checkpoint_sha256": sha256_file(args.output),
            "stop_reason": "target reached" if args.target_score is not None and best_score >= args.target_score else "step budget completed",
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
