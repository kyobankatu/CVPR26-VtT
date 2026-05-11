import importlib
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    print("torch", torch.__version__, flush=True)
    print("torch cuda", torch.version.cuda, flush=True)
    print("cuda available", torch.cuda.is_available(), flush=True)
    print(
        "device",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        flush=True,
    )

    mods = ["torchvision", "selective_scan_cuda", "causal_conv1d_cuda", "VtT"]
    for name in mods:
        print(f"importing {name}", flush=True)
        try:
            importlib.import_module(name)
            print(f"{name} OK", flush=True)
        except Exception as e:
            print(f"{name} NG: {repr(e)}", flush=True)
            raise


if __name__ == "__main__":
    main()
