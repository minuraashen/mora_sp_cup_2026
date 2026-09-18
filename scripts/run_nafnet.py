"""Denoise RGB images offline; save experimental outputs under scripts/runs/nafnet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from nafnet.inference import load_model, read_rgb, sha256_file, to_image
from nafnet.pipeline import restore

SCRIPT_DIR = Path(__file__).resolve().parent


def plan_outputs(
    source: Path, output_dir: Path | None, pattern: str, suffix: str,
    limit: int | None, overwrite: bool, manifest: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Preflight every destination before loading weights or writing any image."""
    if any(character in suffix for character in '/\\:*?"<>|'):
        raise ValueError("Output suffix must be a filename suffix, not a path.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive.")
    if manifest is not None:
        if not source.is_dir():
            raise ValueError("A manifest requires an existing input directory.")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip()]
        names = [record["name"] for record in records]
        if any(Path(name).name != name or "/" in name or "\\" in name for name in names):
            raise ValueError("Manifest image names must not contain directory components.")
        if len(set(names)) != len(names):
            raise ValueError("Manifest contains duplicate image names.")
        inputs = sorted((source / name).resolve() for name in names)
    elif source.is_file():
        inputs = [source.resolve()]
    elif source.is_dir():
        inputs = sorted(path.resolve() for path in source.glob(pattern) if path.is_file())
        # Exclude previous outputs when a broad pattern such as *.png is supplied.
        if suffix:
            inputs = [path for path in inputs if not path.stem.lower().endswith(suffix.lower())]
    else:
        raise FileNotFoundError(f"Input does not exist: {source}")
    if limit:
        inputs = inputs[:limit]
    if not inputs:
        raise ValueError(f"No input images matched {pattern!r} in {source}.")
    sources = set(inputs)
    destinations: set[Path] = set()
    jobs = []
    for path in inputs:
        stem = path.stem[:-6] if path.stem.lower().endswith("_noise") else path.stem
        directory = output_dir.resolve() if output_dir else path.parent
        destination = directory / f"{stem}{suffix}.png"
        if destination.resolve() in sources:
            raise ValueError(f"Refusing to overwrite an input image: {destination}")
        if destination in destinations:
            raise ValueError(f"Multiple inputs would produce the same output: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {destination}. Use --overwrite to replace outputs.")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"Output is not a regular file: {destination}")
        # Catch corrupt/wrong-format images before producing a partial batch.
        if path.exists():
            with Image.open(path) as source_image:
                if source_image.mode != "RGB":
                    raise ValueError(f"{path.name}: expected RGB input, found {source_image.mode}.")
                source_image.verify()
        destinations.add(destination)
        jobs.append((path, destination))
    return jobs


def parser(
    default_suffix: str = "_nafnet",
    default_output_dir: Path | None = SCRIPT_DIR / "runs" / "nafnet",
) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", "--noise_dir", "--input_dir", dest="source", type=Path,
                        required=True, help="One RGB image or a directory of noisy images.")
    result.add_argument("--output_dir", "--denoised_dir", type=Path, default=default_output_dir,
                        help="Output directory (trial default: scripts/runs/nafnet).")
    result.add_argument("--weights", type=Path,
                        default=SCRIPT_DIR / "weights" / "NAFNet-SIDD-width32.pth")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--backend", choices=("pytorch", "directml"), default="pytorch")
    result.add_argument("--onnx-model", type=Path,
                        default=SCRIPT_DIR / "weights/NAFNet-SIDD-width32-256.onnx")
    result.add_argument("--dml-device", type=int, default=0, help="Windows DirectML adapter index.")
    result.add_argument("--tile", type=int, default=256, help="Tile edge in pixels; 0 = full image.")
    result.add_argument("--overlap", type=int, default=32, help="Overlap between adjacent tiles.")
    result.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    result.add_argument("--memory-format", choices=("channels_last", "contiguous"),
                        default="channels_last", help="Tensor storage layout; does not change RGB order.")
    result.add_argument("--preprocess", choices=("none", "adaptive"), default="none",
                        help="Optional blind adaptive-median impulse repair.")
    result.add_argument("--tta", action="store_true", help="Average all 8 D4 transforms; costs about 8x inference.")
    result.add_argument("--chroma-refine", type=float, default=0.0,
                        help="Optional wavelet chroma blend in [0,1]; choose on validation only.")
    result.add_argument("--pattern", default="*_noise.png", help="Directory input glob.")
    result.add_argument("--suffix", default=default_suffix, help="Output filename suffix.")
    result.add_argument("--limit", type=int, help="Process at most this many images.")
    result.add_argument("--manifest", type=Path,
                        help="JSONL input listing; allows inference while the downloader finishes.")
    result.add_argument("--wait-seconds", type=int, default=600,
                        help="Maximum wait per missing manifest input (default: 600).")
    result.add_argument("--overwrite", action="store_true", help="Replace existing outputs, never inputs.")
    result.add_argument("--report", type=Path, help="Optional JSON with settings, timings and SHA-256 hashes.")
    return result


def run(args: argparse.Namespace) -> int:
    if args.tile < 0 or args.overlap < 0 or (args.tile and args.overlap >= args.tile):
        raise ValueError("Require --tile 0 or 0 <= --overlap < --tile.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.wait_seconds < 0:
        raise ValueError("--wait-seconds cannot be negative.")
    if not 0 <= args.chroma_refine <= 1:
        raise ValueError("--chroma-refine must be between zero and one.")
    if args.backend == "directml" and (args.device == "cuda" or args.tile == 0):
        raise ValueError("DirectML uses its own GPU provider and requires a fixed positive tile size.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if args.backend == "directml":
        device_name = "cpu"  # Input/output tensors stay on CPU; ONNX Runtime executes on GPU.
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation. Use --device cpu.")
    jobs = plan_outputs(args.source, args.output_dir, args.pattern, args.suffix,
                        args.limit, args.overwrite, args.manifest)
    protected = {path for pair in jobs for path in pair} | {args.weights.resolve(), args.onnx_model.resolve()}
    if args.report:
        if args.report.resolve() in protected:
            raise ValueError("The report path must differ from images and model weights.")
        if args.report.exists() and not args.overwrite:
            raise FileExistsError(f"Report already exists: {args.report}")
    if any(destination.resolve() in {args.weights.resolve(), args.onnx_model.resolve()}
           for _, destination in jobs):
        raise ValueError("Output path must differ from model weights.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if device_name == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    started = time.perf_counter()
    checkpoint_hash = sha256_file(args.weights)
    backend_details = {}
    if args.backend == "directml":
        from nafnet.directml import DirectMLModel
        model = DirectMLModel(args.onnx_model, checkpoint_hash, args.tile, args.dml_device)
        config = model.config
        backend_details = model.provenance
    else:
        model, config = load_model(args.weights, device)
    memory_format = torch.channels_last if args.memory_format == "channels_last" else torch.contiguous_format
    model = model.to(memory_format=memory_format)
    display_device = f"DirectML GPU {args.dml_device}" if args.backend == "directml" else device_name.upper()
    print(f"NAFNet | width {config['width']} | {display_device} | {len(jobs)} image(s)", flush=True)
    print(f"Tiles: {args.tile or 'full image'} | overlap: {args.overlap} | RGB PNG", flush=True)
    results = []
    for index, (source, destination) in enumerate(jobs, 1):
        deadline = time.monotonic() + args.wait_seconds
        next_notice = 0.0
        while not source.is_file():
            now = time.monotonic()
            if now >= deadline:
                raise FileNotFoundError(f"Timed out waiting for downloaded input: {source}")
            if now >= next_notice:
                print(f"Waiting for download: {source.name}", flush=True)
                next_notice = now + 30
            time.sleep(min(1.0, deadline - now))
        image, (width, height) = read_rgb(source)
        input_hash = sha256_file(source)
        image_started = time.perf_counter()
        last_update = image_started

        def progress(done: int, total: int) -> None:
            nonlocal last_update
            now = time.perf_counter()
            if now - last_update >= 20 and done < total:
                print(f"  {source.name}: tile {done}/{total}", flush=True)
                last_update = now

        prediction, diagnostics = restore(
            model, image, device, args.tile, args.overlap, progress, memory_format,
            args.preprocess, args.tta, args.chroma_refine,
        )
        output_image = to_image(prediction)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation is the default, including if another process wrote it.
        with destination.open("wb" if args.overwrite else "xb") as stream:
            output_image.save(stream, format="PNG")
        elapsed = time.perf_counter() - image_started
        print(f"[{index:02d}/{len(jobs):02d}] {source.name} -> {destination.name}"
              f" | {width}x{height} | {elapsed:.2f}s", flush=True)
        results.append({
            "input": str(source), "output": str(destination),
            "width": width, "height": height, "seconds": round(elapsed, 4),
            "input_sha256": input_hash, "output_sha256": sha256_file(destination),
            "preprocessing": diagnostics,
        })
    elapsed = time.perf_counter() - started
    if args.report:
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": "NAFNet", "architecture": config,
            "checkpoint": str(args.weights.resolve()), "checkpoint_sha256": checkpoint_hash,
            "device": display_device, "backend": args.backend, "backend_details": backend_details,
            "tile": args.tile, "overlap": args.overlap,
            "threads": args.threads, "precision": "float32", "seed": 0,
            "memory_format": args.memory_format,
            "preprocess": args.preprocess, "tta": args.tta, "chroma_refine": args.chroma_refine,
            "signal_processing_versions": {
                "scipy": version("scipy"),
                **({"scikit-image": version("scikit-image"), "PyWavelets": version("PyWavelets")}
                   if args.chroma_refine else {}),
            },
            "torch": torch.__version__, "numpy": np.__version__, "pillow": pillow_version,
            "python": platform.python_version(), "platform": platform.platform(),
            "total_seconds": round(elapsed, 4), "images": results,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(f"Report: {args.report}")
    print(f"Done | {len(results)} saved | {elapsed:.2f}s total")
    return 0


def main(default_suffix: str = "_nafnet") -> int:
    args = parser(default_suffix).parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
