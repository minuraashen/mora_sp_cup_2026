"""Numerical contracts for conditioning and the trainable wavelet architecture."""
import unittest
import numpy as np
import torch
from nafnet.hybrid import condition_image, haar, inverse_haar, upgrade_model
from nafnet.model import NAFNet
from nafnet.inference import predict


class WaveletModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = dict(width=4, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))

    def test_haar_roundtrip_and_energy(self):
        x = torch.randn(2, 5, 20, 24, requires_grad=True)
        bands = haar(x)
        torch.testing.assert_close(inverse_haar(bands), x)
        torch.testing.assert_close(sum(b.square().sum() for b in bands), x.square().sum())
        inverse_haar(bands).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_upgrade_preserves_rgb_prediction_and_trains_gate(self):
        base = NAFNet(**self.config).eval()
        hybrid = upgrade_model(base, self.config, torch.device("cpu"))
        x = torch.rand(1, 10, 33, 35)
        torch.testing.assert_close(hybrid(x), base(x[:, :3]), atol=2e-6, rtol=2e-6)
        hybrid(x).square().mean().backward()
        self.assertGreater(hybrid.wavelet_skips[0].gate.grad.abs().sum().item(), 0)
        self.assertGreater(hybrid.intro.weight.grad[:, 3:].abs().sum().item(), 0)

    def test_conditioning_preserves_observation(self):
        image = np.full((32, 32, 3), 0.2, np.float32)
        image[16, 16] = 1
        result = condition_image(image, {"a": [0.017]*3, "b": [0.006]*3})
        np.testing.assert_array_equal(result[..., :3], image)
        self.assertEqual(result.shape, (32, 32, 10))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[16, 16, 9], 0.9)
        self.assertLess(result[16, 16, 3], 0.3)
        self.assertTrue((result[..., 6:9] > 0).all())

    def test_tiling_accepts_conditioning_and_returns_rgb(self):
        class RGB(torch.nn.Module):
            def forward(self, x):
                return x[:, :3]
        x = torch.rand(1, 10, 40, 44)
        torch.testing.assert_close(predict(RGB(), x, torch.device("cpu"), 24, 8), x[:, :3])


if __name__ == "__main__":
    unittest.main()
