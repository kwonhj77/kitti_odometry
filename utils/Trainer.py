import time
import torch
from torch import nn
from torch.utils.data import DataLoader

# Local imports
from utils.ResultRecorder import DatasetResultRecorder, TestResultRecorder
from utils.Timer import convert_time
from utils.Tester import test

def _train(dataloaders: list[DataLoader], model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    train_recorder = TestResultRecorder(train=True)
    for idx, dataloader in enumerate(dataloaders):
        size = len(dataloader.dataset)
        ds_recorder = DatasetResultRecorder(size=size, num_batches=len(dataloader), dataset_idx=idx, train=True)

        print(f"Train dataset #{idx} ------------")
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            ds_recorder.add_batch_results(loss=loss.detach().clone(), label=y, prediction=pred)

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
        
        ds_recorder.calculate_results(verbose=True)

        train_recorder.append_ds_recorder(ds_recorder)

    train_recorder.calculate_results(verbose=True)

    return train_recorder

def train_and_eval(train_dataloaders: list[DataLoader], test_dataloaders: list[DataLoader], model: nn.Module, loss_fn: nn.Module, device: torch.device, params: dict, run_test: bool):
    print("Training start")
    train_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08)

    train_results = []
    test_results = []

    for epoch in range(params['epochs']):
        print(f"Epoch {epoch+1}\n-------------------------------")
        epoch_start = time.time()
        train_epoch_result = _train(dataloaders=train_dataloaders, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        train_results.append(train_epoch_result)
        if run_test:
            test_epoch_result = test(dataloaders=test_dataloaders, model=model, loss_fn=loss_fn, device=device)
            test_results.append(test_epoch_result)
        epoch_end = time.time() - epoch_start
        print(f"Epoch {epoch+1} Time Elapsed: " + convert_time(epoch_end))

    train_end = time.time() - train_start
    print(f"Time Elapsed during training: " + convert_time(train_end))

    return train_results, test_results