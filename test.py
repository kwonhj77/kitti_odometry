import pykitti
import numpy as np
import matplotlib.pyplot as plt
import torch

def compare_checkpoints(checkpoint_path1, checkpoint_path2):
    """
    Checks if two PyTorch checkpoints have the same weight values.

    Args:
        checkpoint_path1 (str): Path to the first checkpoint file.
        checkpoint_path2 (str): Path to the second checkpoint file.

    Returns:
        bool: True if the checkpoints have the same weights, False otherwise.
    """
    checkpoint1 = torch.load(checkpoint_path1)
    checkpoint2 = torch.load(checkpoint_path2)

    if checkpoint1.keys() != checkpoint2.keys():
        print("Checkpoints have different keys.")
        return False

    issame = True
    for key in checkpoint1:
      if isinstance(checkpoint1[key], dict):
        if checkpoint1[key].keys() != checkpoint2[key].keys():
          print(f"Sub-dictionaries for key '{key}' have different keys.")
          issame = False
        for sub_key in checkpoint1[key]:
          if not torch.equal(checkpoint1[key][sub_key], checkpoint2[key][sub_key]):
            print(f"Values for key '{key}' and sub-key '{sub_key}' are different.")
            issame = False
      elif not torch.equal(checkpoint1[key], checkpoint2[key]):
        print(f"Values for key '{key}' are different.")
        issame = False
    return issame

if __name__ == "__main__":
    # Example usage:
    checkpoint1_path = "./checkpoints/stacked_1024_epoch_1.pt"
    checkpoint2_path = "./checkpoints/stacked_1024_epoch_5.pt"

    # Assuming checkpoint files exist, compare them
    if compare_checkpoints(checkpoint1_path, checkpoint2_path):
        print("Checkpoints have the same weights.")
    else:
        print("Checkpoints have different weights.")



