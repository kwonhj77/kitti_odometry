import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import torch

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from utils.KittiOdomDataset import KittiOdomBatch
from utils.ParamLoader import load_params

# Function to close plot when "Esc" is pressed
def _on_key(event):
    if event.key == "escape":
        plt.close('all')

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

def compare_trajectory():
  sequences_len = [
          4540,
          1100,
          4660,
          800,
          270,
          2760,
          1100,
          1100,
          4070,
          1590,
          1200
  ]
  params = load_params('default')
  dataset_idx = 3
  sequence = KittiOdomBatch(dataset_idx, 0, sequences_len[dataset_idx], params, abs_labels=True)
  rotation = []
  position = []
  for _, _, rot, pos, _, _ in sequence:
    # rot_mat = Rotation.from_euler('XYZ', rot.numpy()).as_matrix().flatten()
    # rotation.append(rot.numpy())
    rotation.append(np.degrees(rot.numpy()))
    position.append(pos.numpy())

  rotation = np.array(rotation)
  position = np.array(position)

  # Load labels derived from .csv
  pos_csv = pd.read_csv('./test_pos.csv').to_numpy()[:,1:]
  rot_csv = pd.read_csv('./test_rot.csv').to_numpy()[:,1:]

  assert np.allclose(pos_csv, position, atol=0.00001)
  assert np.allclose(rot_csv, rotation, atol=0.00001)

  # fig = plt.figure(figsize=(10,6))
  # ax = fig.add_subplot(111, projection='3d')
  # fig.canvas.mpl_connect("key_press_event", _on_key)

  # ax.plot(position[:,0], position[:,1], position[:,2], label='Label Trajectory')
  # ax.plot(pos_csv[:,0], pos_csv[:,1], pos_csv[:,2], label='CSV Trajectory')
  # ax.set_xlabel('X')
  # ax.set_ylabel('Y')
  # ax.set_zlabel('Z')
  # ax.set_title('3D Trajectory')

  # ax.legend()

  # plt.show()
  

     

if __name__ == "__main__":
    # # Example usage:
    # checkpoint1_path = "./checkpoints/stacked_1024_epoch_1.pt"
    # checkpoint2_path = "./checkpoints/stacked_1024_epoch_5.pt"

    # # Assuming checkpoint files exist, compare them
    # if compare_checkpoints(checkpoint1_path, checkpoint2_path):
    #     print("Checkpoints have the same weights.")
    # else:
    #     print("Checkpoints have different weights.")

    compare_trajectory()



