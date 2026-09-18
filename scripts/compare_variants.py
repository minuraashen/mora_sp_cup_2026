"""Run a fixed validation ablation grid, preserving every prediction and score."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def execute(command: list[str], log: Path) -> None:
    """Tee child output to a UTF-8 log, propagating errors immediately."""
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line); stream.flush()
        if process.wait() != 0:
            raise RuntimeError(f"Command failed; inspect {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("scripts/runs/nafnet_docs_20260918"))
    parser.add_argument("--weights", type=Path, default=Path("scripts/weights/NAFNet-Mora-docs-best.pth"))
    parser.add_argument("--onnx-model", type=Path, default=Path("scripts/weights/NAFNet-Mora-docs-best-256.onnx"))
    parser.add_argument("--inference-python", type=Path,
                        default=Path("scripts/.venv-directml/Scripts/python.exe"))
    parser.add_argument("--chroma", action="store_true", help="Also evaluate optional chroma refinement.")
    args = parser.parse_args()
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    manifest = args.run / "validation_inputs.jsonl"
    listing = "".join(json.dumps({"name": f"{image_id}_noise.png"}) + "\n" for image_id in split["validation"])
    if manifest.exists() and manifest.read_text(encoding="utf-8") != listing:
        raise ValueError("Validation manifest disagrees with the fixed split.")
    manifest.write_text(listing, encoding="utf-8")
    variants = {"trained_deep": ["--preprocess", "none"],
                "trained_adaptive": ["--preprocess", "adaptive"],
                "trained_adaptive_tta": ["--preprocess", "adaptive", "--tta"]}
    if args.chroma:
        variants["trained_adaptive_chroma"] = ["--preprocess", "adaptive", "--chroma-refine", "0.15"]
    results = []
    for name, options in variants.items():
        output = args.run / "validation" / name
        report = args.run / f"{name}_run.json"
        metrics = args.run / f"{name}_metrics.json"
        if report.exists() or metrics.exists() or output.exists():
            raise FileExistsError(f"Choose a fresh experiment; candidate already exists: {name}")
        execute([str(args.inference_python), "-u", "scripts/denoise.py",
                 "--noise_dir", "competition_data/public/noisy", "--denoised_dir", str(output),
                 "--weights", str(args.weights), "--onnx-model", str(args.onnx_model),
                 "--backend", "directml", "--threads", "2", "--memory-format", "contiguous",
                 "--manifest", str(manifest), "--wait-seconds", "0", "--report", str(report), *options],
                args.run / f"{name}_inference.log")
        execute([sys.executable, "-u", "scripts/evaluate_run.py", "--pred-dir", str(output),
                 "--split", str(args.run / "split.json"), "--subset", "validation",
                 "--output", str(metrics)], args.run / f"{name}_evaluation.log")
        measured = json.loads(metrics.read_text(encoding="utf-8"))["summary"]
        settings = json.loads(report.read_text(encoding="utf-8"))
        results.append({"name": name, "options": options, "metrics": measured,
                        "total_seconds": settings["total_seconds"]})
    ranked = sorted(results, key=lambda row: row["metrics"]["composite_score"], reverse=True)
    (args.run / "ablation_results.json").write_text(json.dumps({"ranked": ranked}, indent=2) + "\n", encoding="utf-8")
    for row in ranked:
        print(f"{row['name']}: composite {row['metrics']['composite_score']:.8f} | {row['total_seconds']:.2f}s")


if __name__ == "__main__":
    main()
