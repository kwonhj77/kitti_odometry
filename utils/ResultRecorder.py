import numpy as np
import os
import pandas as pd
import warnings

RECORDER_KEYS = ["rot_pred", "pos_pred", "rot_label", "pos_label", "rot_l2_loss", "pos_l2_loss", "rot_diffs", "pos_diffs", "size"]
class ResultRecorder():
    def __init__(self, dataset_idx, train):
        self.dataset_idx = dataset_idx
        if train:
            self.train_or_test = "Train"
        else:
            self.train_or_test = "Test"

        # Sequentially added by add_batch_results
        self.batch_results = {key: list() for key in RECORDER_KEYS}

        # Calculated with calculate_results
        self.dataset_mean_rot_loss = None
        self.dataset_mean_pos_loss = None
        self.dataset_mean_rot_diffs = None
        self.dataset_mean_pos_diffs = None
    
    def add_batch_results(self, loss, label, prediction, batch_size):
        prediction = {key: np.array(value.cpu().detach()) for key,value in prediction.items()}
        label = {key: np.array(value.cpu().detach()) for key,value in label.items()}
        self.batch_results["rot_pred"].append(prediction["rot"])
        self.batch_results["pos_pred"].append(prediction["pos"])
        self.batch_results["rot_label"].append(label["rot"])
        self.batch_results["pos_label"].append(label["pos"])
        self.batch_results["rot_l2_loss"].append(loss["rot"].cpu().detach().numpy())
        self.batch_results["pos_l2_loss"].append(loss["pos"].cpu().detach().numpy())
        self.batch_results["rot_diffs"].append(np.abs(prediction["rot"]-label["rot"]))
        self.batch_results["pos_diffs"].append(np.abs(prediction["pos"]-label["pos"]))
        self.batch_results["size"].append(batch_size)

    def calculate_results(self, verbose):
        self.dataset_mean_rot_loss = np.average(self.batch_results["rot_l2_loss"], weights=self.batch_results["size"])
        self.dataset_mean_pos_loss = np.average(self.batch_results["pos_l2_loss"], weights=self.batch_results["size"])
        mean_rot_diffs_per_batch = []
        mean_pos_diffs_per_batch = []
        for batch_rot_diff, batch_pos_diff in zip(self.batch_results["rot_diffs"], self.batch_results["pos_diffs"]):
            mean_rot_diffs_per_batch.append(np.mean(batch_rot_diff, axis=0))
            mean_pos_diffs_per_batch.append(np.mean(batch_pos_diff, axis=0))
        self.dataset_mean_rot_diffs = np.average(mean_rot_diffs_per_batch, weights=self.batch_results["size"], axis=0)
        self.dataset_mean_pos_diffs = np.average(mean_pos_diffs_per_batch, weights=self.batch_results["size"], axis=0)

        if verbose:
            mean_rot_diffs_str = [f"{d:.4f}" for d in self.dataset_mean_rot_diffs]
            mean_pos_diffs_str = [f"{d:.4f}" for d in self.dataset_mean_pos_diffs]
            print(f"--- \nDataset {self.dataset_idx} {self.train_or_test} Error: \n")
            print(f"Mean Rot Diff: {mean_rot_diffs_str} \n  Mean Pos Diff: {mean_pos_diffs_str} \n")
            print(f"Rot Err: {self.dataset_mean_rot_loss:.6f} \n  Pos Err: {self.dataset_mean_pos_loss:.6f} \n---\n")



    def _to_csv(self, fpath, key):
        assert key in self.batch_results.keys(), f"Invalid key f{key}"
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if key.endswith('l2_loss'):
                columns = ['dataset_idx', 'batch_idx', key, 'size']
                df = pd.DataFrame(columns=columns)

                for idx, (l2_loss, size) in enumerate(zip(self.batch_results[key], self.batch_results['size'])):
                    df = pd.concat([df, pd.DataFrame([[self.dataset_idx, idx, l2_loss, size]], columns=columns)], ignore_index=True)
            else:
                columns = ['dataset_idx', 'batch_idx'] + [f'{i}_{key}' for i in range(len(self.batch_results[key][0][0]))]
                df = pd.DataFrame(columns=columns)

                for idx, val in enumerate(self.batch_results[key]):
                    data = []
                    for i in range(0, val.shape[0]):
                        data.append([self.dataset_idx, idx] + val[i,:].tolist())
                    df = pd.concat([df, pd.DataFrame(data, columns=columns)], ignore_index=True)

            df.to_csv(fpath)
    
    def save_results(self, folder_name, epoch):
        if self.train_or_test == "Train":
            fpath = f'./results/{folder_name}/train'
        else:
            fpath = f'./results/{folder_name}/test'

        subfolders = RECORDER_KEYS[:-1] # exclude size

        if not os.path.exists(fpath):
            os.makedirs(fpath)
            for key in subfolders:
                if not os.path.exists(os.path.join(fpath, key)):
                    os.makedirs(os.path.join(fpath, key))
                if epoch is None:
                    self._to_csv(os.path.join(fpath, key, f'{key}_dataset_{self.dataset_idx}.csv'), key)
                else:
                    self._to_csv(os.path.join(fpath, key, f'{key}_dataset_{self.dataset_idx}_epoch_{epoch}.csv'), key)

    