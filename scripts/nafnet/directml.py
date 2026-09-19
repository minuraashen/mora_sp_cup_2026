"""Optional Windows GPU execution of the same float32 NAFNet graph."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .inference import sha256_file


class DirectMLModel(nn.Module):
    """A fixed-tile ONNX graph with provenance checked against the checkpoint."""

    def __init__(self, path: Path, checkpoint_sha256: str, tile: int, device_id: int = 0):
        super().__init__()
        import onnxruntime as ort

        if device_id < 0:
            raise ValueError("DirectML device index cannot be negative.")
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Install scripts/requirements-directml.txt to use DirectML.")
        if not path.is_file():
            raise FileNotFoundError(f"Export the checkpoint first: missing {path}")
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), sess_options=options,
            providers=[("DmlExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"],
        )
        self.session.disable_fallback()
        if self.session.get_providers()[0] != "DmlExecutionProvider":
            raise RuntimeError("DirectML could not initialize; use the PyTorch CPU backend.")
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("The ONNX export does not match the selected checkpoint.")
        if metadata.get("precision") != "float32" or metadata.get("model") != "NAFNet":
            raise ValueError("Expected a float32 NAFNet export from export_nafnet.py.")
        self.tile = int(metadata["tile"])
        self.input_channels = int(metadata.get("input_channels", 3))
        if tile != self.tile or self.session.get_inputs()[0].shape != [1, self.input_channels, tile, tile]:
            raise ValueError(f"This ONNX export requires --tile {self.tile}.")
        self.config = json.loads(metadata["architecture"])
        self.provenance = {
            "onnx_model": str(path.resolve()), "onnx_sha256": sha256_file(path),
            "onnxruntime": ort.__version__, "directml_device_id": device_id,
            "execution_providers": self.session.get_providers(),
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape) != (1, self.input_channels, self.tile, self.tile):
            raise ValueError(f"DirectML requires {self.tile}-pixel tiles; use PyTorch for smaller images.")
        array = np.ascontiguousarray(image.detach().cpu().numpy(), dtype=np.float32)
        result = self.session.run(["denoised"], {"noisy": array})[0]
        return torch.from_numpy(result)
