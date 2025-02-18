import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation

# Local imports
from utils.ParamLoader import load_params

def _convert_relative_position_to_absolute(rel_pos):
    abs_pos = np.zeros(rel_pos.shape)
    for i in range(len(rel_pos)):
        if i == 0:
            abs_pos[i,:] = rel_pos[i,:]
        else:
            abs_pos[i,:] = abs_pos[i-1,:] + rel_pos[i,:]
    return abs_pos

def _convert_relative_rotation_to_absolute(rel_rot):
    # abs_rot = np.zeros((len(rel_rot), 9))
    abs_rot = np.zeros(rel_rot.shape)
    # Convert to matrix form
    prev_rot_mat = np.eye(3)
    for i in range(len(rel_rot)):
        rot_mat = Rotation.from_euler('XYZ', rel_rot[i,:]).as_matrix()
        abs_rot_mat = np.matmul(rot_mat, prev_rot_mat)
        abs_rot[i,:] = Rotation.from_matrix(abs_rot_mat).as_euler('xyz', degrees=True)
        # abs_rot[i,:] = abs_rot_mat.flatten()
        prev_rot_mat = rot_mat
    return abs_rot

if __name__ == '__main__':
    params = load_params('default')

    last_epoch = 1
    base_dir = './results/quick_stacked/test'
    # Absolute Trajectory Error (ATE) for the position
    pos_label_key = 'pos_label'
    pos_pred_key = 'pos_pred'
    rot_label_key = 'rot_label'
    rot_pred_key = 'rot_pred'

    # Load the data
    rel_pos_label = pd.read_csv(os.path.join(base_dir, pos_label_key, f'{pos_label_key}_epoch_{last_epoch}.csv')).drop(["Unnamed: 0", "batch_idx"], axis=1)
    rel_pos_pred = pd.read_csv(os.path.join(base_dir, pos_pred_key, f'{pos_pred_key}_epoch_{last_epoch}.csv')).drop(["Unnamed: 0", "batch_idx"], axis=1)
    rel_rot_label = pd.read_csv(os.path.join(base_dir, rot_label_key, f'{rot_label_key}_epoch_{last_epoch}.csv')).drop(["Unnamed: 0", "batch_idx"], axis=1)
    rel_rot_pred = pd.read_csv(os.path.join(base_dir, rot_pred_key, f'{rot_pred_key}_epoch_{last_epoch}.csv')).drop(["Unnamed: 0", "batch_idx"], axis=1)

    # Sort by dataset idx
    dataset_indexes = rel_pos_label['dataset_idx'].unique()
    rel_pos_labels = dict.fromkeys(dataset_indexes)
    rel_pos_preds = dict.fromkeys(dataset_indexes)
    rel_rot_labels = dict.fromkeys(dataset_indexes)
    rel_rot_preds = dict.fromkeys(dataset_indexes)

    # Sort by ascending timestamp
    for dataset_idx in dataset_indexes:
        rel_pos_labels[dataset_idx] = rel_pos_label[rel_pos_label['dataset_idx'] == dataset_idx].sort_values(by='timestamp', ascending=True, ignore_index=True)
        rel_pos_preds[dataset_idx] = rel_pos_pred[rel_pos_pred['dataset_idx'] == dataset_idx].sort_values(by='timestamp', ascending=True, ignore_index=True)
        rel_rot_labels[dataset_idx] = rel_rot_label[rel_rot_label['dataset_idx'] == dataset_idx].sort_values(by='timestamp', ascending=True, ignore_index=True)
        rel_rot_preds[dataset_idx] = rel_rot_pred[rel_rot_pred['dataset_idx'] == dataset_idx].sort_values(by='timestamp', ascending=True, ignore_index=True)

    # Convert to numpy arrays
    timestamp = rel_pos_labels[dataset_indexes[0]]['timestamp'].to_numpy()
    for dataset_idx in dataset_indexes:
        rel_pos_labels[dataset_idx] = rel_pos_labels[dataset_idx].iloc[:,-3:].to_numpy()
        rel_pos_preds[dataset_idx] = rel_pos_preds[dataset_idx].iloc[:,-3:].to_numpy()
        rel_rot_labels[dataset_idx] = rel_rot_labels[dataset_idx].iloc[:,-3:].to_numpy()
        rel_rot_preds[dataset_idx] = rel_rot_preds[dataset_idx].iloc[:,-3:].to_numpy()

    # Unnormalize values
    unnormalize_pos = lambda x : (x * params['pos_std']) + params['pos_mean']
    unnormalize_rot = lambda x : (x * params['rot_std']) + params['rot_mean']
    for dataset_idx in dataset_indexes:
        rel_pos_labels[dataset_idx] = np.apply_along_axis(unnormalize_pos, 1, rel_pos_labels[dataset_idx])
        rel_pos_preds[dataset_idx] = np.apply_along_axis(unnormalize_pos, 1, rel_pos_preds[dataset_idx])
        
        rel_rot_labels[dataset_idx] = np.apply_along_axis(unnormalize_rot, 1, rel_rot_labels[dataset_idx])
        rel_rot_preds[dataset_idx] = np.apply_along_axis(unnormalize_rot, 1, rel_rot_preds[dataset_idx])

    # Compute absolute position and rotation values
    abs_pos_labels = dict.fromkeys(dataset_indexes)
    abs_pos_preds = dict.fromkeys(dataset_indexes)
    abs_rot_labels = dict.fromkeys(dataset_indexes)
    abs_rot_preds = dict.fromkeys(dataset_indexes)

    for dataset_idx in dataset_indexes:
        abs_pos_labels[dataset_idx] = _convert_relative_position_to_absolute(rel_pos_labels[dataset_idx])
        abs_pos_preds[dataset_idx] = _convert_relative_position_to_absolute(rel_pos_preds[dataset_idx])

        abs_rot_labels[dataset_idx] = _convert_relative_rotation_to_absolute(rel_rot_labels[dataset_idx])
        abs_rot_preds[dataset_idx] = _convert_relative_rotation_to_absolute(rel_rot_preds[dataset_idx])

    pd.DataFrame(abs_pos_labels[dataset_indexes[0]]).to_csv('./test_pos.csv')
    pd.DataFrame(abs_rot_labels[dataset_indexes[0]]).to_csv('./test_rot.csv')

    print("Exiting...")