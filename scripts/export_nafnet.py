"""Export a verified NAFNet checkpoint for optional DirectML acceleration."""

import argparse
import json
from pathlib import Path

import torch

from nafnet.inference import load_model, sha256_file

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/NAFNet-SIDD-width32.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/NAFNet-SIDD-width32-256.onnx")
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()
    if args.tile < 16 or args.tile % 16:
        parser.error("--tile must be a positive multiple of 16.")
    if args.output.exists():
        parser.error(f"Export already exists: {args.output}")
    import onnx

    torch.set_num_threads(2)
    model, config = load_model(args.weights, torch.device("cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, model.intro.in_channels, args.tile, args.tile)
    print(f"Exporting NAFNet, tile {args.tile}, opset 17...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            model, sample, str(args.output), input_names=["noisy"], output_names=["denoised"],
            opset_version=17, dynamo=False, external_data=False,
        )
    graph = onnx.load(args.output)
    onnx.helper.set_model_props(graph, {
        "checkpoint_sha256": sha256_file(args.weights),
        "architecture": json.dumps(config), "tile": str(args.tile),
        "model": "NAFNet", "precision": "float32",
        "input_channels": str(model.intro.in_channels),
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, args.output)
    print(f"Saved: {args.output}\nSHA-256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
