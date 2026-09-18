"""Verify a completed inference report against its images and original inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--download-report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=460)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    original = json.loads(args.download_report.read_text(encoding="utf-8"))
    original_hashes = {row["name"]: row["sha256"] for row in original["images"]}
    rows = report["images"]
    if len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} outputs, found {len(rows)}.")
    inputs = [Path(row["input"]) for row in rows]
    outputs = [Path(row["output"]) for row in rows]
    if len(set(inputs)) != len(rows) or len(set(outputs)) != len(rows):
        raise ValueError("The report contains duplicate inputs or outputs.")
    if {path.name for path in inputs} != set(original_hashes):
        raise ValueError("Inference inputs differ from the verified download manifest.")
    if digest(Path(report["checkpoint"])) != report["checkpoint_sha256"]:
        raise ValueError("The checkpoint changed after inference.")
    details = report.get("backend_details", {})
    if "onnx_model" in details and digest(Path(details["onnx_model"])) != details["onnx_sha256"]:
        raise ValueError("The exported model changed after inference.")
    for row, source, output in zip(rows, inputs, outputs):
        if digest(source) != row["input_sha256"] or row["input_sha256"] != original_hashes[source.name]:
            raise ValueError(f"Original input changed: {source}")
        if digest(output) != row["output_sha256"]:
            raise ValueError(f"Output differs from the recorded prediction: {output}")
        with Image.open(output) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (row["width"], row["height"]):
                raise ValueError(f"Invalid output format or dimensions: {output}")
            image.verify()
    print(f"Verified {len(rows)} RGB PNG outputs, dimensions, and output/model hashes.")
    print(f"All {len(rows)} original inputs match their download hashes.")


if __name__ == "__main__":
    main()
