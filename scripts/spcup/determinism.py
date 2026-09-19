"""
determinism.py -- must be imported and configured BEFORE torch.

The submission is disqualified if the frozen code does not reproduce the submitted
PNGs, so the strongest defence is that the inference pipeline contains no stochasticity
at all: no random crops, no dropout, no sampling, no test-time randomness of any kind.
The seeds below are belt-and-braces, not the mechanism.

Two environment variables genuinely must precede the torch import (CUBLAS workspace
config and the thread counts), which is why this lives in its own module rather than
at the top of denoise.py.
"""

from __future__ import annotations

import os

_CONFIGURED = False


def configure(seed: int = 0, threads: int | None = None) -> int:
    """Set env vars, seeds and torch flags. Returns the thread count actually used."""
    global _CONFIGURED

    if threads is None or threads <= 0:
        threads = min(8, os.cpu_count() or 4)

    # These must be set before torch loads its backends.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False          # benchmark picks kernels non-deterministically
    torch.backends.cuda.matmul.allow_tf32 = False   # TF32 silently changes results on Ampere+
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    _CONFIGURED = True
    return threads


def is_configured() -> bool:
    return _CONFIGURED
