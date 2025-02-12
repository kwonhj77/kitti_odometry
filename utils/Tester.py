import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List

from utils.ResultRecorder import ResultRecorder

@torch.no_grad()  # Disable gradient calculation to save compute
def test(dataloaders: List[DataLoader], model: nn.Module, loss_fn: nn.Module, device: torch.device):
    
    model.eval()

    test_results = dict()

    for _, dataloader in enumerate(dataloaders):
        assert len(dataloader) == 1, "only 1 batch should be in each dataloader!"
        dataset_idx = dataloader.dataset.raw_dataset.sequence

        if dataset_idx not in test_results:
            test_results[dataset_idx] = ResultRecorder(dataset_idx=dataset_idx, train=False)
        
        batch_size = len(dataloader.dataset)

        X_prev, X_curr, rot, pos = next(iter(dataloader))
        X_prev, X_curr, rot, pos = X_prev.to(device), X_curr.to(device), rot.to(device), pos.to(device)

        # Compute prediction error
        pred_rot, pred_pos = model(X_prev, X_curr)
        
        loss_rot = loss_fn(pred_rot, rot)
        loss_pos = loss_fn(pred_pos, pos)

        # Store results
        label = {
            "rot": rot,
            "pos": pos,
        }
        pred = {
            "rot": pred_rot,
            "pos": pred_pos,
        }

        loss = {
            "rot": loss_rot.detach().clone(),
            "pos": loss_pos.detach().clone(),
        }
        
        test_results[dataset_idx].add_batch_results(loss=loss, label=label, prediction=pred, batch_size=batch_size)
    
    for idx, recorder in test_results.items():
        recorder.calculate_results(verbose=True)
    
    return test_results