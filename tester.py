import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.ResultRecorder import DatasetResultRecorder, TestResultRecorder

@torch.no_grad()  # Disable gradient calculation to save compute
def test(dataloaders: list[DataLoader], model: nn.Module, loss_fn: nn.Module, device: torch.device):

    test_recorder = TestResultRecorder(train=True)
    
    for idx, dataloader in enumerate(dataloaders):
        ds_recorder = DatasetResultRecorder(size=len(dataloader.dataset), num_batches=len(dataloader), dataset_idx=idx, train=True)

        model.eval()

        for (X,y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            ds_recorder.add_batch_results(loss=loss_fn(pred,y), label=y, prediction=pred)

        ds_recorder.calculate_results(verbose=True)

        test_recorder.append_ds_recorder(ds_recorder)

    test_recorder.calculate_results(verbose=True)

    return test_recorder
