import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

@torch.no_grad()  # Disable gradient calculation to save compute
def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, diffs = 0, [0 for _ in range(12)]
    for value in dataloader:
        X, y = value['image'].to(device), value['odom'].to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y)
        diff = np.mean([label.cpu().numpy() - prediction.cpu().numpy() for label, prediction in zip(y, pred)], axis=0)
        diffs = [x + y for x, y in zip(diff, diffs)]
    test_loss /= num_batches
    diffs = [(d / size).item() for d in diffs]
    diffs_str = [f"{d:.2f}" for d in diffs]
    print(f"Test Error: \n Difference: {diffs_str}%, Avg loss: {test_loss:.8f} \n")