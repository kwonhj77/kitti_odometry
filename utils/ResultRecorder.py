import numpy as np
import os
import pandas as pd
import warnings

PER_FRAME_KEYS = ["rot_pred", "pos_pred", "rot_label", "pos_label", "rot_diffs", "pos_diffs"]
PER_BATCH_KEYS = ["rot_l2_loss", "pos_l2_loss"]
INDEX_KEYS = ["dataset_idx", "batch_idx", "timestamp"]
RECORDER_KEYS =  PER_FRAME_KEYS + PER_BATCH_KEYS + INDEX_KEYS + ["size",]

class ResultRecorder():
    def __init__(self, train):
        if train:
            self.train_or_test = "Train"
        else:
            self.train_or_test = "Test"

        # Sequentially added by add_batch_results
        self.batch_results = {key: list() for key in RECORDER_KEYS}

        # Calculated with calculate_results
        self.result_mean_rot_loss = None
        self.result_mean_pos_loss = None
        self.result_mean_rot_diffs = None
        self.result_mean_pos_diffs = None
    
    def add_batch_results(self, loss, label, prediction, dataset_idx, batch_idx, timestamp):
        prediction = {key: np.array(value.cpu().detach()) for key,value in prediction.items()}
        label = {key: np.array(value.cpu().detach()) for key,value in label.items()}

        # per batch items
        self.batch_results["rot_l2_loss"].append(loss["rot"].cpu().detach().numpy())
        self.batch_results["pos_l2_loss"].append(loss["pos"].cpu().detach().numpy())
        self.batch_results["dataset_idx"].append(dataset_idx)
        self.batch_results["batch_idx"].append(batch_idx)

        # per frame items
        self.batch_results["rot_pred"].append(prediction["rot"])
        self.batch_results["pos_pred"].append(prediction["pos"])
        self.batch_results["rot_label"].append(label["rot"])
        self.batch_results["pos_label"].append(label["pos"])
        self.batch_results["rot_diffs"].append(np.abs(prediction["rot"]-label["rot"]))
        self.batch_results["pos_diffs"].append(np.abs(prediction["pos"]-label["pos"]))
        self.batch_results["timestamp"].append(timestamp)

        # Used for calculate_results
        self.batch_results["size"].append(len(timestamp))

    def calculate_results(self, verbose):
        self.result_mean_rot_loss = np.average(self.batch_results["rot_l2_loss"], weights=self.batch_results["size"])
        self.result_mean_pos_loss = np.average(self.batch_results["pos_l2_loss"], weights=self.batch_results["size"])
        mean_rot_diffs_per_batch = []
        mean_pos_diffs_per_batch = []
        for batch_rot_diff, batch_pos_diff in zip(self.batch_results["rot_diffs"], self.batch_results["pos_diffs"]):
            mean_rot_diffs_per_batch.append(np.mean(batch_rot_diff, axis=0))
            mean_pos_diffs_per_batch.append(np.mean(batch_pos_diff, axis=0))
        self.result_mean_rot_diffs = np.average(mean_rot_diffs_per_batch, weights=self.batch_results["size"], axis=0)
        self.result_mean_pos_diffs = np.average(mean_pos_diffs_per_batch, weights=self.batch_results["size"], axis=0)

        if verbose:
            mean_rot_diffs_str = [f"{d:.4f}" for d in self.result_mean_rot_diffs]
            mean_pos_diffs_str = [f"{d:.4f}" for d in self.result_mean_pos_diffs]
            print(f"--- \n{self.train_or_test} Error: \n")
            print(f"Mean Rot Diff: {mean_rot_diffs_str} \n  Mean Pos Diff: {mean_pos_diffs_str} \n")
            print(f"Rot Err: {self.result_mean_rot_loss:.6f} \n  Pos Err: {self.result_mean_pos_loss:.6f} \n---\n")



    def _to_csv(self, fpath, key):
        assert key in self.batch_results.keys(), f"Invalid key f{key}"
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if key in PER_BATCH_KEYS:
                columns = ['dataset_idx', 'batch_idx', key]
                df = pd.DataFrame(columns=columns)

                for i in range(len(self.batch_results[key])):
                    dataset_idx = self.batch_results['dataset_idx'][i]
                    batch_idx = self.batch_results['batch_idx'][i]
                    l2_loss = self.batch_results[key][i]
                    df = pd.concat([df, pd.DataFrame([[dataset_idx, batch_idx, l2_loss]], columns=columns)], ignore_index=True)
            elif key in PER_FRAME_KEYS:
                columns = ['dataset_idx', 'batch_idx', 'timestamp'] + [f'{i}_{key}' for i in range(len(self.batch_results[key][0][0]))]
                df = pd.DataFrame(columns=columns)

                for i, val in enumerate(self.batch_results[key]):
                    dataset_idx = self.batch_results['dataset_idx'][i]
                    batch_idx = self.batch_results['batch_idx'][i]
                    data = []
                    for j in range(val.shape[0]):
                        data.append([dataset_idx, batch_idx, self.batch_results['timestamp'][i][j]] + val[j,:].tolist())
                    df = pd.concat([df, pd.DataFrame(data, columns=columns)], ignore_index=True)

            df.to_csv(fpath)
    
    def save_results(self, folder_name, epoch):
        if self.train_or_test == "Train":
            fpath = f'./results/{folder_name}/train'
        else:
            fpath = f'./results/{folder_name}/test'

        subfolders = PER_FRAME_KEYS + PER_BATCH_KEYS

        if not os.path.exists(fpath):
            os.makedirs(fpath)
        for key in subfolders:
            if not os.path.exists(os.path.join(fpath, key)):
                os.makedirs(os.path.join(fpath, key))
            if epoch is None:
                self._to_csv(os.path.join(fpath, key, f'{key}.csv'), key)
            else:
                self._to_csv(os.path.join(fpath, key, f'{key}_epoch_{epoch}.csv'), key)

    