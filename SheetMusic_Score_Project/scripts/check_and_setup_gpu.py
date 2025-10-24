"""
Check torch/torchvision/torchaudio CUDA availability. On Windows, print guidance to
install CUDA-enabled wheels if a GPU is available but torch reports no CUDA.
Optionally pass --install to attempt pip installation automatically.
"""

from __future__ import annotations
import argparse
import subprocess
import sys


def check_env() -> dict:
    import torch
    import torchvision
    import torchaudio

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info


def try_install_cuda_wheels() -> int:
    # Default to CUDA 12.4 wheels for PyTorch 2.4+ (adjust as needed)
    cmds = [
        [sys.executable, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu124",
         "torch", "torchvision", "torchaudio"],
    ]
    for cmd in cmds:
        print(">>>", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            return ret.returncode
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="Attempt to install CUDA-enabled torch wheels")
    args = ap.parse_args()

    try:
        info = check_env()
    except Exception as e:
        print(f"Failed to import torch stack: {e}")
        sys.exit(1)

    print("Environment:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if not info["cuda_available"]:
        print("CUDA not available in current torch build.")
        if args.install:
            print("Attempting to install CUDA-enabled wheels (cu124)...")
            code = try_install_cuda_wheels()
            sys.exit(code)
        else:
            print("Hint: Run with --install to attempt CUDA wheel installation on Windows.")
    else:
        print("CUDA is available. You're good to go!")


if __name__ == "__main__":
    main()
