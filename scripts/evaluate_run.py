"""Strict all-image or split evaluation using the unchanged official scorer."""

import argparse
import csv
import json
from pathlib import Path

from nafnet.inference import sha256_file
from nafnet.scoring import official, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("competition_data/public"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("scripts/runs/nafnet_docs_20260918/split.json"))
    parser.add_argument("--subset", choices=("all", "train", "validation"), default="validation")
    parser.add_argument("--output", type=Path, required=True, help="JSON path; CSV and text are saved alongside.")
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    ids = sorted(split["train"] + split["validation"]) if args.subset == "all" else split[args.subset]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in evaluation split.")
    for image_id in ids:
        for path in (args.data / "noisy" / f"{image_id}_noise.png",
                     args.data / "ground_truth" / f"{image_id}.png", args.pred_dir / f"{image_id}.png"):
            if not path.is_file():
                raise FileNotFoundError(f"Refusing partial-set evaluation: {path}")
    if any(args.output.with_suffix(suffix).exists() for suffix in (".json", ".csv", ".txt")):
        raise FileExistsError("Choose a fresh metrics output path.")
    rows = []
    for index, image_id in enumerate(ids, 1):
        row = official.evaluate_image(image_id, args.pred_dir / f"{image_id}.png",
                                      args.data / "ground_truth" / f"{image_id}.png",
                                      args.data / "noisy" / f"{image_id}_noise.png")
        rows.append(row)
        if index % 25 == 0:
            print(f"Scored {index}/{len(ids)}", flush=True)
    summary = summarize(rows)
    payload = {"subset": args.subset, "summary": summary, "images": rows,
               "split_sha256": sha256_file(args.split),
               "evaluator_sha256": sha256_file(Path(official.__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    message = (f"Images scored:      {len(rows)} ({args.subset})\n"
               f"Mean PSNR:          {summary['psnr']:.4f} dB\n"
               f"Mean SSIM:          {summary['ssim']:.6f}\n"
               f"Mean Delta PSNR:    {summary['delta_psnr']:+.4f} dB\n"
               f"Mean Delta SSIM:    {summary['delta_ssim']:+.6f}\n"
               f"Composite Score:    {summary['composite_score']:.8f}\n")
    args.output.with_suffix(".txt").write_text(message, encoding="utf-8")
    print(message)


if __name__ == "__main__":
    main()
