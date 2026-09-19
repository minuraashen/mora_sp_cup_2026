"""Offline inference for the trained, ten-channel wavelet NAFNet."""
import argparse
import json
from pathlib import Path
import platform
import time
import torch
import numpy as np
import scipy
import PIL
from nafnet.hybrid import condition_image
from nafnet.inference import load_model, predict, read_rgb, sha256_file, to_image
from run_nafnet import plan_outputs

SCRIPTS = Path(__file__).resolve().parent


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--noise_dir", "--input_dir", "--input", dest="source", type=Path, required=True)
    cli.add_argument("--denoised_dir", "--output_dir", dest="output", type=Path, required=True)
    cli.add_argument("--weights", type=Path, default=SCRIPTS / "weights/Wavelet-NAFNet-Mora-best.pth")
    cli.add_argument("--device", choices=("auto", "cpu", "cuda", "directml"), default="auto")
    cli.add_argument("--backend", choices=("pytorch", "directml"), default="pytorch",
                     help="directml uses ONNX; device directml uses native PyTorch DML.")
    cli.add_argument("--onnx-model", type=Path, default=SCRIPTS / "weights/Wavelet-NAFNet-Mora-256.onnx")
    cli.add_argument("--tile", type=int, default=256)
    cli.add_argument("--overlap", type=int, default=32)
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--report", type=Path,
                     help="Defaults to a sibling of the output directory, keeping images separate.")
    cli.add_argument("--resume", action="store_true", help="Reuse only outputs verified by this run's journal.")
    return cli


def run(args):
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Input and output directories must differ.")
    if args.threads < 1:
        raise ValueError("Thread count must be positive.")
    if args.report is None:
        args.report = args.output.parent / (args.output.name + "_run.json")
    jobs = plan_outputs(args.source, args.output, "*_noise.png", "", args.limit,
                        args.resume, args.manifest)
    if args.report.exists() and not args.resume:
        raise FileExistsError(f"Report already exists: {args.report}")
    if args.report.resolve() in {path.resolve() for pair in jobs for path in pair}:
        raise ValueError("Report path collides with an image.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260919)
    torch.use_deterministic_algorithms(True)
    digest = sha256_file(args.weights)
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    fit = payload.get("noise_model")
    if not fit or payload["params_ema"]["intro.weight"].shape[1] != 10:
        raise ValueError("A trained ten-channel checkpoint with noise calibration is required.")
    del payload
    if args.backend == "directml":
        from nafnet.directml import DirectMLModel
        model = DirectMLModel(args.onnx_model, digest, args.tile)
        device = torch.device("cpu")
    else:
        selected = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if selected == "directml":
            import torch_directml
            device = torch_directml.device()
        else:
            device = torch.device(selected)
        model, _ = load_model(args.weights, device)
    provenance = {"checkpoint_sha256": digest, "backend": args.backend, "device": str(device),
                  "tile": args.tile, "overlap": args.overlap, "threads": args.threads,
                  "torch": str(torch.__version__), "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__,
                  "implementation_sha256": sha256_file(SCRIPTS / "nafnet/hybrid.py"),
                  "inference_sha256": sha256_file(SCRIPTS / "nafnet/inference.py"),
                  "backbone_sha256": sha256_file(SCRIPTS / "nafnet/model.py"),
                  "classical_sha256": sha256_file(SCRIPTS / "nafnet/classical.py"),
                  "runner_sha256": sha256_file(Path(__file__))}
    if args.backend == "directml":
        provenance.update(model.provenance)
    journal = args.report.with_suffix(".jsonl")
    existing = {}
    if args.resume and journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["provenance"] != provenance:
                raise ValueError("Resume journal belongs to a different model or inference configuration.")
            existing[record["name"]] = record
    elif journal.exists():
        raise FileExistsError(f"Journal exists; use --resume: {journal}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results, started = [], time.perf_counter()
    execution_label = "ONNX DirectML GPU" if args.backend == "directml" else str(device)
    print(f"Wavelet NAFNet | {len(jobs)} images | {execution_label}", flush=True)
    for index, (source, target) in enumerate(jobs, 1):
        source_hash = sha256_file(source)
        if target.exists():
            row = existing.get(source.name)
            if not row or row["input_sha256"] != source_hash or row["output_sha256"] != sha256_file(target):
                raise ValueError(f"Cannot reuse unverified or changed output: {target}")
        else:
            beginning = time.perf_counter()
            observation, size = read_rgb(source)
            array = observation[0].permute(1, 2, 0).numpy()
            conditioned = torch.from_numpy(condition_image(array, fit)).permute(2, 0, 1).unsqueeze(0)
            result = predict(model, conditioned, device, args.tile, args.overlap)
            image = to_image(result)
            if image.size != size:
                raise ValueError("Output dimensions changed.")
            temporary = target.with_suffix(".partial")
            image.save(temporary, format="PNG")
            temporary.replace(target)
            row = {"name": source.name, "output": str(target.resolve()),
                   "input_sha256": source_hash, "output_sha256": sha256_file(target),
                   "size": list(size), "seconds": time.perf_counter() - beginning,
                   "provenance": provenance}
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        results.append(row)
        if index % 10 == 0 or index == len(jobs):
            print(f"[{index}/{len(jobs)}] {source.name} -> {target.name} | {time.perf_counter()-started:.1f}s", flush=True)
    report = {"provenance": provenance, "count": len(results), "images": results,
              "elapsed_seconds": time.perf_counter() - started}
    temporary = args.report.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return report


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
