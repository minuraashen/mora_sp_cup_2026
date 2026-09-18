"""Numerical checks for the hybrid pipeline and augmentation assumptions."""

from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nafnet.augmentation import inverse_transform, synthesize, transform
from nafnet.classical import estimate_noise, refine_chroma, repair_impulses
from nafnet.losses import restoration_loss
from nafnet.pipeline import restore


class HybridTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("skimage") and importlib.util.find_spec("pywt"),
                         "optional wavelet dependencies are not installed")
    def test_chroma_refinement_preserves_flat_rgb_without_nan(self):
        image = np.full((32, 32, 3), 0.3, np.float32)
        result = refine_chroma(image, 0.15)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, image, atol=1e-7)

    def test_d4_inverts_all_eight_rectangular_transforms(self):
        image = np.arange(7 * 11 * 3).reshape(7, 11, 3)
        for index in range(8):
            np.testing.assert_array_equal(inverse_transform(transform(image, index), index), image)
        square = image[:7, :7]
        self.assertEqual(len({transform(square, index).tobytes() for index in range(8)}), 8)

    def test_tta_preserves_identity_with_tiling(self):
        torch.manual_seed(19)
        image = torch.rand(1, 3, 29, 43)
        output, _ = restore(torch.nn.Identity(), image, torch.device("cpu"),
                            tile=16, overlap=4, tta=True)
        torch.testing.assert_close(output, image, atol=3e-7, rtol=1e-6)

    def test_adaptive_filter_repairs_an_isolated_hot_pixel(self):
        image = np.full((31, 31, 3), 0.25, np.float32)
        image[15, 15] = 1
        result, details = repair_impulses(image)
        np.testing.assert_allclose(result, 0.25)
        self.assertGreater(details["repaired_fraction"], 0)
        np.testing.assert_array_equal(image[15, 15], np.ones(3))

    def test_adaptive_filter_preserves_a_clean_step_edge(self):
        image = np.full((32, 32, 3), 0.1, np.float32)
        image[:, 16:] = 0.8
        result, _ = repair_impulses(image)
        np.testing.assert_array_equal(result, image)

    def test_blind_estimator_tracks_gaussian_noise(self):
        rng = np.random.default_rng(2)
        image = rng.normal(0.5, 0.04, (256, 256, 3)).astype(np.float32)
        np.testing.assert_allclose(estimate_noise(image), 0.04, atol=0.002)

    def test_noise_generator_is_seeded_and_signal_dependent(self):
        fit = {"training_sigma": [0.02], "median_sigma": 0.02,
               "a": [0.001] * 3, "b": [0.00001] * 3}
        clean = np.full((256, 256, 3), 0.1, np.float32)
        clean[:, 128:] = 0.5
        first = synthesize(clean, fit, np.random.default_rng(9))
        second = synthesize(clean, fit, np.random.default_rng(9))
        np.testing.assert_array_equal(first, second)
        residual = first - clean
        variance_low = np.var(np.clip(residual[:, :128], -0.08, 0.08))
        variance_high = np.var(np.clip(residual[:, 128:], -0.08, 0.08))
        self.assertGreater(variance_high, variance_low * 2)

    def test_loss_is_lower_for_clean_predictions_and_has_finite_gradients(self):
        torch.manual_seed(8)
        clean = torch.rand(1, 3, 32, 32)
        noisy = (clean + torch.randn_like(clean) * 0.05).requires_grad_()
        loss = restoration_loss(noisy, clean)
        self.assertLess(float(restoration_loss(clean, clean)), float(loss.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(noisy.grad).all())
        self.assertGreater(float(noisy.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
