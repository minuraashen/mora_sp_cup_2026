"""
infer.py -- model loading and the forward pass (whole-image, tiled, and D4 TTA).

Tiling is deliberately NOT the default. 992 = 2^5 * 31 is a multiple of 32, so a
4-level NAFNet needs zero padding at full resolution, and the measured peak working
set is ~1.2 GB -- comfortable on a 6 GB GPU and on any mid-range CPU. Tiling would
also process 1.3-2.4x more pixels (slower, against the runtime criterion) and adds a
second code path that can fail. It is kept only as an automatic OOM fallback, where
being slower is obviously preferable to crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import nafnet


def load_model(weights_path, device="cpu", tlc_kernel: int = 0):
    """Load a checkpoint written by train.py or export_model.py.

    Accepts fp16-stored weights (cast up to fp32) so the shipped checkpoint can be
    half the size without changing inference numerics meaningfully.
    """
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "w32")
    sd = ck.get("model", ck.get("params", ck.get("ema", ck)))
    sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}

    model = nafnet.build(arch)
    model.load_state_dict(sd, strict=True)
    if tlc_kernel and tlc_kernel > 0:
        nafnet.convert_to_tlc(model, tlc_kernel)
    model.eval().to(device)
    return model, arch


def _forward_whole(model, x):
    return model(x).clamp_(0.0, 1.0)


def _cosine_ramp(n: int, dtype, device):
    if n <= 0:
        return torch.ones(0, dtype=dtype, device=device)
    t = torch.arange(n, dtype=dtype, device=device)
    return 0.5 - 0.5 * torch.cos(torch.pi * (t + 1.0) / (n + 1.0))


def _blend_window(h: int, w: int, ov: int, dtype, device):
    """Separable raised-cosine window: ramps over the overlap, flat in the interior."""
    def axis(n):
        v = torch.ones(n, dtype=dtype, device=device)
        m = min(ov, n // 2)
        if m > 0:
            r = _cosine_ramp(m, dtype, device)
            v[:m] = r
            v[n - m:] = r.flip(0)
        return v
    return axis(h)[:, None] * axis(w)[None, :]


def _forward_tiled(model, x, tile: int, overlap: int = 64, halo: int = 64):
    """Overlap-add tiled forward with a partition-of-unity guarantee.

    Each tile is extracted with `halo` pixels of real context on every side (reflected
    at the image border) which is cropped off before blending -- the halo exists only
    so the receptive field sees valid data. Tile starts are spread with linspace so
    every overlap is the same width; the naive range(0, W, stride) leaves a degenerate
    sliver at the right and bottom edges, which is the classic source of seams.

    The accumulated weight is divided out explicitly rather than assuming the window
    sums to one. That makes the reconstruction exact for ANY tile geometry.
    """
    _, _, H, W = x.shape
    dtype, device = x.dtype, x.device

    def starts(total):
        """Evenly spaced tile origins covering [0, total).

        linspace rather than range(0, total, stride): even spacing makes every overlap
        the same width, whereas a fixed stride leaves a narrow sliver at the right and
        bottom edges, which is the usual source of visible seams. n (not n+1) points
        is the minimum count that still covers the axis at >= the requested overlap --
        using n+1 silently processes an extra row and column of tiles.
        """
        if tile >= total:
            return [0]
        n = max(2, int(np.ceil((total - overlap) / (tile - overlap))))
        return [int(round(v)) for v in np.linspace(0, total - tile, n)]

    ys, xs = starts(H), starts(W)
    acc = torch.zeros((x.shape[0], 3, H, W), dtype=torch.float32, device=device)
    wsum = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)

    for y0 in ys:
        for x0 in xs:
            th, tw = min(tile, H - y0), min(tile, W - x0)
            ey0, ex0 = y0 - halo, x0 - halo
            ey1, ex1 = y0 + th + halo, x0 + tw + halo
            pt, pl = max(0, -ey0), max(0, -ex0)
            pb, pr = max(0, ey1 - H), max(0, ex1 - W)
            crop = x[:, :, max(0, ey0):min(H, ey1), max(0, ex0):min(W, ex1)]
            if pt or pl or pb or pr:
                crop = torch.nn.functional.pad(crop, (pl, pr, pt, pb), mode="reflect")

            out = _forward_whole(model, crop)
            out = out[:, :, halo:halo + th, halo:halo + tw].float()

            win = _blend_window(th, tw, overlap, torch.float32, device)[None, None]
            acc[:, :, y0:y0 + th, x0:x0 + tw] += out * win
            wsum[:, :, y0:y0 + th, x0:x0 + tw] += win

    if float(wsum.min()) <= 0:
        raise RuntimeError("tiling left uncovered pixels")
    return (acc / wsum).clamp_(0.0, 1.0).to(dtype)


# D4 group. Each entry is (forward, inverse) acting on a (N,C,H,W) tensor.
_D4 = (
    (lambda t: t,                              lambda t: t),
    (lambda t: torch.flip(t, [3]),             lambda t: torch.flip(t, [3])),
    (lambda t: torch.flip(t, [2]),             lambda t: torch.flip(t, [2])),
    (lambda t: torch.flip(t, [2, 3]),          lambda t: torch.flip(t, [2, 3])),
    (lambda t: torch.rot90(t, 1, [2, 3]),      lambda t: torch.rot90(t, -1, [2, 3])),
    (lambda t: torch.rot90(t, 3, [2, 3]),      lambda t: torch.rot90(t, -3, [2, 3])),
    (lambda t: torch.rot90(t, 1, [2, 3]).flip([3]), lambda t: torch.rot90(t.flip([3]), -1, [2, 3])),
    (lambda t: torch.rot90(t, 3, [2, 3]).flip([3]), lambda t: torch.rot90(t.flip([3]), -3, [2, 3])),
)


def run(model, x, tile: int = 0, tta: int = 1, overlap: int = 64, halo: int = 64):
    """Forward pass with optional self-ensemble.

    TTA averages in float64 over a FIXED transform order and quantises only once at
    the very end -- averaging already-quantised uint8 outputs would throw away most
    of the benefit and make the result depend on rounding order.
    """
    n_t = 1 if tta <= 1 else (4 if tta <= 4 else 8)
    acc = None
    for fwd, inv in _D4[:n_t]:
        xt = fwd(x).contiguous()
        if tile and tile > 0:
            yt = _forward_tiled(model, xt, tile, overlap, halo)
        else:
            yt = _forward_whole(model, xt)
        y = inv(yt).double()
        acc = y if acc is None else acc + y
    return (acc / n_t).clamp_(0.0, 1.0)


def forward_with_oom_ladder(model, x, tile: int = 0, tta: int = 1, verbose=False):
    """Try the requested geometry, then progressively smaller tiles on memory failure."""
    ladder = [tile] if tile and tile > 0 else [0, 512, 256]
    if tile and tile > 0:
        ladder = [tile, 256]
    last = None
    for t in ladder:
        try:
            return run(model, x, tile=t, tta=tta), t
        except (RuntimeError, MemoryError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, MemoryError):
                raise
            last = exc
            if x.device.type == "cuda":
                torch.cuda.empty_cache()
            if verbose:
                print(f"    [oom] tile={t} failed, retrying smaller", flush=True)
    raise RuntimeError(f"all tile sizes exhausted: {last}")
