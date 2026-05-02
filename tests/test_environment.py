"""Smoke tests verifying the environment is correctly configured for training."""
import torch


def test_torch_imports():
    assert torch.__version__


def test_cuda_available():
    assert torch.cuda.is_available(), (
        "CUDA not available — verify NVIDIA driver and PyTorch CUDA build"
    )


def test_gpu_is_4090():
    name = torch.cuda.get_device_name(0)
    assert "4090" in name, f"Unexpected GPU: {name}"


def test_bf16_supported():
    assert torch.cuda.is_bf16_supported(), "bf16 not supported on this device"


def test_tensor_op_on_gpu():
    x = torch.randn(8, 8, device="cuda")
    y = x @ x.T
    assert y.shape == (8, 8)
    assert y.device.type == "cuda"
