import time
import torch
from torch import nn
from torch.utils.data import DataLoader

# Local imports
from utils.ResultRecorder import ResultRecorder
from utils.Timer import convert_time
from utils.Tester import test

def _train(dataloaders: list[DataLoader], model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):

    model.train()

    train_results = {}
    num_batches = len(dataloaders)
    for batch, dataloader in enumerate(dataloaders):
        assert len(dataloader) == 1, "only 1 batch should be in each dataloader!"
        dataset_idx = dataloader.dataset.raw_dataset.sequence

        if dataset_idx not in train_results:
            train_results[dataset_idx] = ResultRecorder(dataset_idx=dataset_idx, train=True)

        batch_size = len(dataloader.dataset)

        X, rot, pos = next(iter(dataloader))
        X, rot, pos = X.to(device), rot.to(device), pos.to(device)

        # Compute prediction error
        pred_rot, pred_pos = model(X)
        
        loss_rot = loss_fn(pred_rot, rot)
        loss_pos = loss_fn(pred_pos, pos)

        # Backprop
        torch.autograd.backward(loss_rot, loss_pos)
        # loss_rot.backward()
        # loss_pos.backward()
        optimizer.step()
        optimizer.zero_grad()

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

        train_results[dataset_idx].add_batch_results(loss=loss, label=label, prediction=pred, batch_size=batch_size)

        if batch % 10 == 0:
            print(f"Train L2 rotation err: {loss_rot.item():>7f} - L2 position err: {loss_pos.item():>7f} - [batch {batch:>3d}/{num_batches:>4d}]\n")
    
    for idx, recorder in train_results.items():
        recorder.calculate_results(verbose=True)

    return train_results

def train_and_eval(train_dataloaders: list[DataLoader], test_dataloaders: list[DataLoader], model: nn.Module, loss_fn: nn.Module, device: torch.device, params: dict, run_test: bool):
    print("################## Training start ##################")
    train_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08)

    # Freeze Resnet backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.gru.parameters():
        param.requires_grad = True
    for param in model.rot_head.parameters():
        param.requires_grad = True
    for param in model.pos_head.parameters():
        param.requires_grad = True
    
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