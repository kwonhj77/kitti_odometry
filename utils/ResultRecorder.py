import numpy as np
import pandas as pd

class DatasetResultRecorder():
    def __init__(self, size, num_batches, dataset_idx, train):
        self.size = size
        self.num_batches = num_batches
        self.dataset_idx = dataset_idx
        if train:
            self.train_or_test = "Train"
        else:
            self.train_or_test = "Test"

        # Sequentially added by add_batch_results
        self.batch_diffs = []
        self.batch_losses = []

        # Calculated with calculate_results
        self.dataset_mean_loss = None
        self.dataset_mean_diffs = None
    
    def add_batch_results(self, loss, label, prediction):
        diff = np.mean(np.abs(np.array(label.cpu().detach()) - np.array(prediction.cpu().detach())), axis=0)
        self.batch_diffs.append(diff)
        self.batch_losses.append(loss.cpu().detach().numpy())

    def calculate_results(self, verbose):
        self.dataset_mean_loss = sum(self.batch_losses) / self.num_batches
        self.dataset_mean_diffs = np.mean(self.batch_diffs, axis=0)

        if verbose:
            mean_diff_str = [f"{d:.4f}" for d in self.dataset_mean_diffs]
            print(f"{self.train_or_test} Dataset #{self.dataset_idx} Error: \n Mean Average Error: {mean_diff_str}, L2: {self.dataset_mean_loss:.8f} \n")


class TestResultRecorder():
    def __init__(self, train):
        if train:
            self.train_or_test = "Train"
        else:
            self.train_or_test = "Test"

        # Sequentially added by append_ds_recorder
        self.size, self.num_batches = 0, 0
        self.batch_losses, self.batch_diffs = [], []
        self.ds_recorders = []

        # Calculated with calculate_results
        self.mean_loss, self.mean_diffs = 0, [0 for _ in range(12)]

    
    def append_ds_recorder(self, ds_recorder: DatasetResultRecorder):
        self.size += ds_recorder.size
        self.num_batches += ds_recorder.num_batches
        self.batch_losses.extend(ds_recorder.batch_losses)
        self.batch_diffs.extend(ds_recorder.batch_diffs)
        self.ds_recorders.append(ds_recorder)
    
    def calculate_results(self, verbose):
        self.mean_loss = (sum(self.batch_losses) / self.num_batches)
        diff_sums = np.sum(self.batch_diffs, axis=0)
        self.mean_diffs = [(d / self.size).item() for d in diff_sums]

        if verbose:
            mean_diffs_str = [f"{d:.4f}" for d in self.mean_diffs]
            print(f"--- Total {self.train_or_test} Error: \n Mean Absolute Error: {mean_diffs_str}, L2 loss: {self.mean_loss:.8f} ---\n")

    def to_csv(self, fpath: str):
        columns = ['dataset_idx','l2_loss',] + [f'{i}_MAE' for i in range(1,13)]
        df = pd.DataFrame(columns=columns)
        for idx, rec in enumerate(self.ds_recorders):
            for loss, diff in zip(rec.batch_losses, rec.batch_diffs):
                row = [idx, loss] + diff.tolist()
                df = pd.concat([df, pd.DataFrame([row], columns=columns)], ignore_index=True)

        df.to_csv(fpath)