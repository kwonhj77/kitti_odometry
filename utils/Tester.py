import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.ResultRecorder import ResultRecorder

@torch.no_grad()  # Disable gradient calculation to save compute
def test(dataloaders: list[DataLoader], model: nn.Module, loss_fn: nn.Module, device: torch.device):
    
    model.eval()

    test_results = dict()

    for _, dataloader in enumerate(dataloaders):
        assert len(dataloader) == 1, "only 1 batch should be in each dataloader!"
        dataset_idx = dataloader.dataset.dataset.sequence

        if dataset_idx not in test_results:
            test_results[dataset_idx] = ResultRecorder(dataset_idx=dataset_idx, train=False)
        
        batch_size = len(dataloader.dataset)

        X, y = next(iter(dataloader))
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        test_results[dataset_idx].add_batch_results(loss=loss.detach().clone(), label=y, prediction=pred, batch_size=batch_size)
    
    for idx, recorder in test_results.items():
        recorder.calculate_results(verbose=True)
    
    return test_results



    
    # for idx, dataloader in enumerate(dataloaders):
    #     ds_recorder = DatasetResultRecorder(size=len(dataloader.dataset), num_batches=len(dataloader), dataset_idx=dataloader.dataset.dataset.sequence, train=False)

    #     for (X,y) in dataloader:
    #         X, y = X.to(device), y.to(device)
    #         pred = model(X)
    #         ds_recorder.add_batch_results(loss=loss_fn(pred,y), label=y, prediction=pred)

    #     ds_recorder.calculate_results(verbose=True)

    #     test_recorder.append_ds_recorder(ds_recorder)

    # test_recorder.calculate_results(verbose=True)

    # return test_recorder
