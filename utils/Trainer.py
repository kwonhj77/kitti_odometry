import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

# Local imports
from utils.ResultRecorder import ResultRecorder
from utils.Timer import convert_time
from utils.Tester import test

def _train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, device: torch.device):

    model.train()

    train_recorder = ResultRecorder(train=True)
    num_batches = len(dataloader)
    for batch_idx, (X_prev, X_curr, rot, pos, seq, timestamp) in enumerate(dataloader):
        X_prev, X_curr, rot, pos = X_prev.to(device), X_curr.to(device), rot.to(device), pos.to(device)

        # Compute prediction error
        pred_rot, pred_pos = model(X_prev, X_curr)

        loss_rot = loss_fn(pred_rot, rot)
        loss_pos = loss_fn(pred_pos, pos)

        total_loss = loss_rot + loss_pos
        
        # Backprop
        total_loss.backward()
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

        train_recorder.add_batch_results(loss=loss, label=label, prediction=pred, dataset_idx=seq, batch_idx=batch_idx, timestamp=timestamp)

        if batch_idx % 10 == 0:
            print(f"Train L2 rotation err: {loss_rot.item():>7f} - L2 position err: {loss_pos.item():>7f} - [batch {batch_idx:>3d}/{num_batches:>4d}]\n")    

    train_recorder.calculate_results(verbose=True)

    return train_recorder

def train_and_eval(train_dataloader: DataLoader, test_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, early_stopping_patience: int, best_val_loss: float, device: torch.device, params: dict, save_checkpoint: str, save_results: str):
    print("################## Training start ##################")
    train_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    for epoch in range(params['epochs']):
        print(f"Epoch {epoch+1}\n-------------------------------")
        epoch_start = time.time()
        train_epoch_recorder = _train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, device=device)
        print("Evaluating epoch...")
        test_epoch_recorder = test(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
        val_loss = test_epoch_recorder.result_mean_rot_loss+test_epoch_recorder.result_mean_pos_loss
        print(f"Validation loss: {val_loss:.6f}\n")

        scheduler.step(val_loss)
        epoch_end = time.time() - epoch_start
        print(f"Epoch {epoch+1} Time Elapsed: " + convert_time(epoch_end) + "\n")

        # Save epoch checkpoint and results
        if save_checkpoint:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            fpath = f'./checkpoints/{save_checkpoint}_epoch_{epoch+1}' + '.pt'
            print("Saving model checkpoint to: \n" + fpath)
            torch.save(model.state_dict(), fpath)
        
        if save_results:
            print(f"Saving epoch results to ./results/{save_results}")
            train_epoch_recorder.save_results(folder_name=save_results, epoch=epoch+1)
            test_epoch_recorder.save_results(folder_name=save_results, epoch=epoch+1)
        
        # Early stopping - currently will not trigger until a good best_val_loss is found.
        early_stopping_patience = 5
        best_val_loss = float('inf') if best_val_loss is None else best_val_loss
        epochs_no_improve = 0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        

    train_end = time.time() - train_start
    print(f"Time Elapsed during training: " + convert_time(train_end))

    return