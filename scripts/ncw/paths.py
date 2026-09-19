"""Repository paths, resolved relative to this file."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPTS_DIR.parent

ANALYSIS_DIR = SCRIPTS_DIR / "analysis" / "noise_characterization"
WEIGHTS_DIR = SCRIPTS_DIR / "weights"
ARTIFACTS_DIR = SCRIPTS_DIR / "artifacts"

PUBLIC_NOISY = REPO_ROOT / "competition_data" / "public" / "noisy"
PUBLIC_GT = REPO_ROOT / "competition_data" / "public" / "ground_truth"
SUBMISSION_NOISY = REPO_ROOT / "competition_data" / "submissions" / "noisy"
SUBMISSION_DENOISED = REPO_ROOT / "competition_data" / "submissions" / "denoised"

OFFICIAL_EVALUATOR = REPO_ROOT / "evaluation" / "evaluate.py"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "ncw_1h_best_composite_fp16.pth"

N_PUBLIC = 460
IMG_SIZE = 992
NOISE_SUFFIX = "_noise"
