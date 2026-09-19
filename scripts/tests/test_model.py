"""Shape and invariant checks for NCW-NAFNet that need no checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ncw.conditioning import impulse_feature
from ncw.model import NCWNAFNet
from ncw.wavelet import haar_dwt, haar_pyramid


def test_haar_is_orthonormal() -> None:
    x = torch.randn(1, 3, 16, 16)
    ll, lh, hl, hh = haar_dwt(x)
    energy_in = x.pow(2).sum()
    energy_out = sum(band.pow(2).sum() for band in (ll, lh, hl, hh))
    assert torch.allclose(energy_in, energy_out, rtol=1e-5), "Haar must preserve energy"


def test_pyramid_shapes() -> None:
    bands = haar_pyramid(torch.randn(1, 3, 64, 64), 4)
    assert [b.shape[1] for b in bands] == [9, 9, 9, 9]
    assert [b.shape[-1] for b in bands] == [32, 16, 8, 4]


def test_impulse_feature_detects_a_spike() -> None:
    x = torch.full((1, 3, 9, 9), 0.5)
    quiet = impulse_feature(x)
    assert quiet.max().item() == 0.0, "flat input must produce no impulse response"
    x[0, :, 4, 4] = 1.0
    spike = impulse_feature(x)
    assert spike[0, 0, 4, 4].item() == 1.0, "isolated spike must saturate the detector"


def test_zero_gates_reproduce_the_backbone() -> None:
    """The defining property: an untrained wrapper is an exact identity."""
    torch.manual_seed(0)
    model = NCWNAFNet(width=8, middle_blk_num=1,
                      enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1)).eval()
    x = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        wrapped = model(x)
        plain = model.backbone(x)
    assert torch.equal(wrapped, plain), "zero-gated wrapper must equal the backbone"


def test_output_shape_survives_odd_sizes() -> None:
    model = NCWNAFNet(width=8, middle_blk_num=1,
                      enc_blk_nums=(1, 1, 1, 1), dec_blk_nums=(1, 1, 1, 1)).eval()
    x = torch.rand(1, 3, 37, 53)
    with torch.no_grad():
        assert model(x).shape == x.shape


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
    raise SystemExit(1 if failures else 0)
