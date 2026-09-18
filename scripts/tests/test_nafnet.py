"""Offline checks for normalization, tiling and safe output planning."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nafnet.inference import load_model, predict, to_image
from nafnet.model import LayerNorm2d, NAFNet
from run_nafnet import plan_outputs


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_layer_norm_matches_pytorch_reference(self):
        x = torch.randn(2, 6, 13, 17)
        layer = LayerNorm2d(6)
        expected = F.layer_norm(x.permute(0, 2, 3, 1), (6,), eps=1e-6)
        torch.testing.assert_close(layer(x), expected.permute(0, 3, 1, 2))

    def test_tiles_cover_odd_and_narrow_images_without_dark_edges(self):
        for height, width in ((79, 103), (1, 79), (67, 1), (15, 19)):
            with self.subTest(size=(height, width)):
                image = torch.rand(1, 3, height, width)
                output = predict(nn.Identity(), image, torch.device("cpu"), tile=32, overlap=8)
                torch.testing.assert_close(output, image, rtol=1e-5, atol=1e-6)

    def test_channels_last_tiling_preserves_pixels(self):
        image = torch.rand(1, 3, 79, 103)
        output = predict(nn.Identity(), image, torch.device("cpu"), tile=32, overlap=8,
                         memory_format=torch.channels_last)
        torch.testing.assert_close(output, image, rtol=1e-5, atol=1e-6)

    def test_network_preserves_arbitrary_dimensions(self):
        model = NAFNet(width=4, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))
        with torch.inference_mode():
            result = model(torch.rand(1, 3, 31, 47))
        self.assertEqual(tuple(result.shape), (1, 3, 31, 47))
        self.assertTrue(torch.isfinite(result).all())

    def test_missing_weight_is_rejected_instead_of_using_random_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_model(Path(directory) / "missing.pth", torch.device("cpu"))

    def test_nonfinite_output_is_rejected(self):
        with self.assertRaises(ValueError):
            to_image(torch.full((1, 3, 4, 4), float("nan")))

    def test_rgb_order_and_rounding(self):
        pixel = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1)
        result = to_image(pixel)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (0, 128, 255))


class OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "001_noise.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(self.source)

    def test_output_is_beside_input_and_excluded_on_rerun(self):
        jobs = plan_outputs(self.root, None, "*.png", "_nafnet", None, False)
        self.assertEqual(jobs[0][1], self.root.resolve() / "001_nafnet.png")
        with Image.open(self.source) as source:
            source.save(jobs[0][1])
        with self.assertRaises(FileExistsError):
            plan_outputs(self.root, None, "*.png", "_nafnet", None, False)
        rerun = plan_outputs(self.root, None, "*.png", "_nafnet", None, True)
        self.assertEqual(len(rerun), 1)

    def test_input_cannot_be_overwritten_even_with_overwrite_flag(self):
        with self.assertRaises(ValueError):
            plan_outputs(self.source, None, "*.png", "_noise", None, True)

    def test_colliding_names_are_rejected(self):
        with Image.open(self.source) as source:
            source.save(self.root / "001_noise.jpg")
        with self.assertRaises(ValueError):
            plan_outputs(self.root, None, "*", "_nafnet", None, False)

    def test_no_matching_images_is_an_error(self):
        with self.assertRaises(ValueError):
            plan_outputs(self.root, None, "*.tif", "_nafnet", None, False)

    def test_manifest_can_plan_pending_downloads(self):
        manifest = self.root / "inputs.jsonl"
        manifest.write_text(json.dumps({"name": "002_noise.png"}) + "\n")
        jobs = plan_outputs(self.root, self.root / "outputs", "*.png", "_nafnet", None,
                            False, manifest)
        self.assertEqual(jobs[0][0].name, "002_noise.png")
        self.assertEqual(jobs[0][1].name, "002_nafnet.png")

    def test_manifest_cannot_escape_input_directory(self):
        manifest = self.root / "inputs.jsonl"
        manifest.write_text(json.dumps({"name": "../outside_noise.png"}) + "\n")
        with self.assertRaises(ValueError):
            plan_outputs(self.root, None, "*.png", "_nafnet", None, False, manifest)


if __name__ == "__main__":
    unittest.main()
