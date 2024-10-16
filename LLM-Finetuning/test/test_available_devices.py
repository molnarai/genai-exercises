#!/usr/bin/env python3
# https://developer.apple.com/metal/pytorch/
import torch

def test_mps():
    """
    Test if an MPS device is available and print a tensor on it.
    """
    # # Check if an MPS device is available
    # if torch.backends.mps.is_available():
    #     print ("MPS device found.")
    # else:
    #     print ("MPS device not found.")

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")


if __name__ == "__main__":
    test_mps()
