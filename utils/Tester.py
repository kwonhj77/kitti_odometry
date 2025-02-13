import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.ResultRecorder import ResultRecorder

@torch.no_grad()  # Disable gradient calculation to save compute
def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device):
    
    model.eval()

    test_recorder = ResultRecorder(train=False)

    num_batches = len(dataloader)
    for batch_idx, (X_prev, X_curr, rot, pos, seq, timestamp) in enumerate(dataloader):
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

        test_recorder.add_batch_results(loss=loss, label=label, prediction=pred, dataset_idx=seq, batch_idx=batch_idx, timestamp=timestamp)

        if batch_idx % 10 == 0:
            print(f"Testing batch {batch_idx:>3d}/{num_batches:>4d}")
    test_recorder.calculate_results(verbose=True)

    return test_recorder