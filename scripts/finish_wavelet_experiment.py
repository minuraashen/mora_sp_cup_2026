"""Export, infer, score, and verify the finished hybrid experiment."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from PIL import Image
from nafnet.inference import sha256_file
from nafnet.scoring import official, summarize
from run_wavelet_experiment import ROOT, RUN, run_logged

INFERENCE = ROOT / "scripts/.venv-directml/Scripts/python.exe"
WEIGHTS = ROOT / "scripts/weights/Wavelet-NAFNet-Mora-best.pth"
ONNX = ROOT / "scripts/weights/Wavelet-NAFNet-Mora-256.onnx"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_outputs(label, ids):
    folder = RUN / label
    expected = {f"{i:03d}.png" for i in ids}
    if {p.name for p in folder.iterdir()} != expected:
        raise ValueError(f"Unexpected or missing outputs in {folder}")
    report = read(RUN / f"{label}_run.json")
    if report["count"] != len(expected) or report["provenance"]["checkpoint_sha256"] != sha256_file(WEIGHTS):
        raise ValueError("Output provenance mismatch.")
    for row in report["images"]:
        path = Path(row["output"])
        if path.parent.resolve() != folder.resolve() or sha256_file(path) != row["output_sha256"]:
            raise ValueError(f"Invalid output checksum or path: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.format != "PNG" or image.size != (992, 992):
                raise ValueError(f"Invalid image: {path}")
            image.verify()
    return report


def verify_originals():
    total = 0
    for split, kind in (("public", "noisy"), ("public", "ground_truth"), ("submissions", "noisy")):
        report = read(ROOT / f"scripts/runs/{split}_{kind}_download.json")
        expected_count = 20 if split == "submissions" else 460
        if len(report["images"]) != expected_count or report["failures"]:
            raise ValueError("Incomplete source manifest.")
        directory = ROOT / "competition_data" / split / kind
        for row in report["images"]:
            if sha256_file(directory / row["name"]) != row["sha256"]:
                raise ValueError(f"Original input changed: {row['name']}")
            total += 1
    return total


def inference(label, source, extra=()):
    run_logged([str(INFERENCE), "-u", "scripts/denoise.py", "--noise_dir", str(source),
        "--denoised_dir", str(RUN / label), "--report", str(RUN / f"{label}_run.json"),
        "--weights", str(WEIGHTS), "--onnx-model", str(ONNX), "--resume", *extra], f"{label}.log")


def evaluate():
    path = RUN / "public_metrics.json"
    digest = sha256_file(WEIGHTS)
    if path.exists():
        result = read(path)
        if result["checkpoint_sha256"] != digest:
            raise ValueError("Metrics belong to a different checkpoint.")
        return result
    split = read(RUN / "split.json")
    rows = []
    for index in range(1, 461):
        image_id = f"{index:03d}"
        rows.append(official.evaluate_image(image_id, RUN / "public" / f"{image_id}.png",
            ROOT / "competition_data/public/ground_truth" / f"{image_id}.png",
            ROOT / "competition_data/public/noisy" / f"{image_id}_noise.png"))
        if index % 25 == 0:
            print(f"Official evaluation {index}/460", flush=True)
    validation = [row for row in rows if row["image_id"] in split["validation"]]
    training = [row for row in rows if row["image_id"] in split["train"]]
    if len(validation) != 50 or len(training) != 410:
        raise ValueError("Evaluation split is invalid.")
    result = {"all_public": summarize(rows), "validation": summarize(validation),
        "training": summarize(training), "images": rows, "checkpoint_sha256": digest,
        "evaluator_sha256": sha256_file(Path(official.__file__)),
        "split_sha256": sha256_file(RUN / "split.json")}
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(path, result)
    return result


def main():
    finished = read(RUN / "training_finished.json")
    if finished["checkpoint_sha256"] != sha256_file(WEIGHTS):
        raise ValueError("Training checkpoint changed after completion.")
    if not ONNX.exists():
        run_logged([str(INFERENCE), "scripts/export_nafnet.py", "--weights", str(WEIGHTS),
                    "--output", str(ONNX)], "export.log")
    inference("submissions", ROOT / "competition_data/submissions/noisy", ("--backend", "directml"))
    verify_outputs("submissions", range(461, 481))
    inference("public", ROOT / "competition_data/public/noisy", ("--backend", "directml"))
    public = verify_outputs("public", range(1, 461))
    metrics = evaluate()
    # Explicitly test offline CPU fallback on three severity-representative images.
    manifest = RUN / "cpu_inputs.jsonl"
    manifest.write_text("".join(json.dumps({"name": name}) + "\n" for name in
        ("196_noise.png", "343_noise.png", "074_noise.png")), encoding="utf-8")
    inference("cpu_check", ROOT / "competition_data/public/noisy",
              ("--device", "cpu", "--threads", "8", "--manifest", str(manifest)))
    inference("repeat_check", ROOT / "competition_data/submissions/noisy/461_noise.png",
              ("--backend", "directml"))
    if sha256_file(RUN / "repeat_check/461.png") != sha256_file(RUN / "submissions/461.png"):
        raise ValueError("GPU repeatability check failed.")
    originals = verify_originals()
    archive = RUN / "test_outputs_461-480.zip"
    with zipfile.ZipFile(archive.with_suffix(".partial"), "w", zipfile.ZIP_DEFLATED) as stream:
        for index in range(461, 481):
            stream.write(RUN / "submissions" / f"{index}.png", f"{index}.png")
    archive.with_suffix(".partial").replace(archive)
    cpu = read(RUN / "cpu_check_run.json")
    verification = {"public_images": public["count"], "test_images": 20,
        "original_images_unchanged": originals, "rgb_png_size": [992, 992],
        "same_backend_repeat_sha256_match": True, "cpu_images": cpu["count"],
        "cpu_mean_seconds": sum(row["seconds"] for row in cpu["images"]) / cpu["count"],
        "checkpoint_sha256": sha256_file(WEIGHTS), "onnx_sha256": sha256_file(ONNX),
        "zip_sha256": sha256_file(archive)}
    write(RUN / "verification.json", verification)
    baseline = read(ROOT / "scripts/runs/nafnet_docs_20260918/classical_comparison.json")["starter_validation"]["summary"]
    lines = ["# Wavelet NAFNet results", "", "Training completed: " + str(finished["completed_steps"]) + " additional steps.",
        "", "| Metric | Starter, validation 50 | Hybrid, validation 50 | Hybrid, public 460 |",
        "|---|---:|---:|---:|"]
    for key in ("psnr", "ssim", "delta_psnr", "delta_ssim", "composite_score"):
        lines.append(f"| {key} | {baseline[key]:.8f} | {metrics['validation'][key]:.8f} | {metrics['all_public'][key]:.8f} |")
    lines += ["", "The 50 validation images select the checkpoint; all-public metrics include 410 training images.",
        "Test images 461-480 have hidden ground truth; no test PSNR/SSIM is claimed.", "",
        "Outputs: `public/001.png` through `460.png`; `submissions/461.png` through `480.png`.",
        "Archive: `test_outputs_461-480.zip` (exactly 20 PNGs).", "",
        f"CPU fallback: {verification['cpu_mean_seconds']:.2f} seconds/image, measured on three images.",
        "Checkpoint SHA-256: `" + sha256_file(WEIGHTS) + "`.",
        "ONNX SHA-256: `" + sha256_file(ONNX) + "`.", "",
        "All 940 original input/target files verified unchanged; GPU repeat output is byte-identical."]
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = ROOT / "scripts/README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("The 5,000-step hybrid experiment is running.",
                        "The 5,000-step hybrid experiment has completed training and output verification.")
    readme.write_text(text, encoding="utf-8")
    print("\n".join(lines), flush=True)
    # The PowerShell script validates paths and replacement provenance before deleting old predictions.
    run_logged(["powershell", "-NoProfile", "-File", "scripts/replace_old_nafnet_outputs.ps1"], "replacement.log")


if __name__ == "__main__":
    main()
