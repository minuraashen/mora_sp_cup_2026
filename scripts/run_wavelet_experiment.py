"""Run resumable hybrid training, then verified output generation automatically."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/runs/wavelet_nafnet_20260919"


def run_logged(command, name):
    with (RUN / name).open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finish-only", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "job_status.json").write_text(json.dumps({"status": "running", "started_utc":
        datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    try:
        if not args.finish_only and not (RUN / "training_finished.json").exists():
            command = [sys.executable, "-u", "scripts/train_nafnet.py", "--hybrid",
                "--run", str(RUN), "--weights", "scripts/weights/NAFNet-Mora-docs-best.pth",
                "--output", "scripts/weights/Wavelet-NAFNet-Mora-best.pth", "--device", "directml",
                "--steps", "5000", "--batch", "1", "--patch-size", "256",
                "--synthetic-probability", "0.25", "--ema-decay", "0.99", "--lr", "0.00002",
                "--validate-every", "500", "--checkpoint-every", "100", "--seed", "20260919"]
            if args.resume:
                command += ["--resume", str(RUN / "training_state.pth")]
            run_logged(command, "training.log")
        run_logged([sys.executable, "-u", "scripts/finish_wavelet_experiment.py"], "completion.log")
    except Exception as error:
        (RUN / "job_status.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2), encoding="utf-8")
        raise
    (RUN / "job_status.json").write_text(json.dumps({"status": "complete", "finished_utc":
        datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
