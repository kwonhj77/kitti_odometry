import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.KittiOdomDataset import get_dataloader
from utils.ParamLoader import load_params

PARAM_FILE = "stacked_1024"

def get_norm_values(dataloader):
    all_pos = []
    all_rot = []
    for _, _, rot, pos, _, _ in dataloader:
        all_pos.append(pos)
        all_rot.append(rot)

    all_pos = torch.cat(all_pos, dim=0).numpy()
    all_rot = torch.cat(all_rot, dim=0).numpy()

    pos_mean = np.mean(all_pos, axis=0)
    pos_std = np.std(all_pos, axis=0)
    rot_mean = np.mean(all_rot, axis=0)
    rot_std = np.std(all_rot, axis=0)

    return pos_mean, pos_std, rot_mean, rot_std




if __name__ == '__main__':
    params = load_params(PARAM_FILE)
    keys_to_remove = ['pos_mean', 'pos_std', 'rot_mean', 'rot_std']
    for key in keys_to_remove:
        del params[key]

    dataloader = get_dataloader(sequences=params['train_sequences'], params=params, shuffle=False)

    pos_mean, pos_std, rot_mean, rot_std = get_norm_values(dataloader)
    with open('./pose_norm_values.txt', 'w') as f:
        f.write(f"pos_mean: {pos_mean}\n")
        f.write(f"pos_std: {pos_std}\n")
        f.write(f"rot_mean: {rot_mean}\n")
        f.write(f"rot_std: {rot_std}\n")
    print("Exiting...")