"""Require CPU/native-DirectML forward and gradient parity before training."""

from copy import deepcopy
import argparse
from pathlib import Path

import torch
import torch_directml

from nafnet.inference import load_model
from nafnet.losses import restoration_loss
from nafnet.model import NAFNet
from nafnet.hybrid import HybridNAFNet, upgrade_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    device = torch_directml.device()
    model, config = load_model(Path("scripts/weights/NAFNet-SIDD-width32.pth"), torch.device("cpu"))
    if args.hybrid:
        model = upgrade_model(model, config, torch.device("cpu")).eval()
    channels = 10 if args.hybrid else 3
    sample = torch.rand(1, channels, 128, 128)
    expected = model(sample)
    actual = model.to(device)(sample.to(device)).cpu()
    difference = float((actual - expected).abs().max())
    print(f"Pretrained CPU/DirectML maximum output difference: {difference:.8f}")
    if difference > 1e-3:
        raise AssertionError("Native DirectML forward values differ from CPU.")
    del model, expected, actual
    factory = HybridNAFNet if args.hybrid else NAFNet
    small = factory(width=8, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1))
    for name, parameter in small.named_parameters():
        if name.endswith(("beta", "gamma")):
            parameter.data.fill_(0.1)
        if name.endswith("gate"):
            parameter.data.fill_(0.1)
    accelerated = deepcopy(small).to(device)
    sample, target = torch.rand(2, channels, 32, 32), torch.rand(2, 3, 32, 32)
    cpu_loss = restoration_loss(small(sample), target)
    gpu_loss = restoration_loss(accelerated(sample.to(device)), target.to(device))
    cpu_loss.backward(); gpu_loss.backward()
    gradient_error = max(float((cpu.grad - gpu.grad.cpu()).abs().max())
                         for cpu, gpu in zip(small.parameters(), accelerated.parameters()))
    print(f"Loss difference: {abs(float(cpu_loss.detach()) - float(gpu_loss.detach().cpu())):.8f}")
    print(f"Maximum parameter-gradient difference: {gradient_error:.8f}")
    if gradient_error > 2e-3:
        raise AssertionError("Native DirectML gradients differ from CPU.")
    print("Training backend parity passed.")


if __name__ == "__main__":
    main()
