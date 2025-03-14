import argparse
import os
import time
import torch
from torch import nn
from torchsummary import summary

# Force utf-8 encoding to resolve issues printing to terminal
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# Local imports
from models.KittiOdomNN import KittiOdomNN
from utils.KittiOdomDataset import get_dataloader
from utils.DeviceLoader import get_device
from utils.ParamLoader import load_params
from utils.Timer import convert_time
from utils.Trainer import train_and_eval
from utils.Tester import test
from utils.Visualizer import generate_plots

def get_argparser():
    # Parse input arguments
    parser = argparse.ArgumentParser(
        prog='Kitti Odometery NN',
        description='Training & Evaluator for the Kitti Odometery dataset'
    )
    parser.add_argument('--train', action='store_true', help='Add arg to train. Otherwise uses .pt specified in --checkpoint')
    parser.add_argument('--save_results', type=str, default=None, help='Saves results to .csv files under a subdirectory. If ommitted, will not save results.')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Local path to .pt which contains model checkpoint')
    parser.add_argument('--save_checkpoint', type=str, default=None, help='Local directory to save model checkpoints. If ommitted, will not save checkpoint.')
    parser.add_argument('--params', type=str, default=None, help='Yaml filename under ./checkpoints which contains hyperparameters to tune. If ommitted, will use default params from default.yaml')

    args = parser.parse_args()
    assert not (args.train and args.load_checkpoint), "Selecting both training and --load_checkpoint is not supported"
    return args


def main():
    exec_start_time = time.time()
    args = get_argparser()
    params = load_params(args.params)

    device = get_device()
    input_dims = [params['img_size'] for _ in range(2)]
    model = KittiOdomNN(gru_hidden_size=params['gru_hidden_size'], device=device).to(device)
    # Freeze pretrained backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Explicitly unfreeze other layers
    for param in model.gru.parameters():
        param.requires_grad = True
    for param in model.rot_head.parameters():
        param.requires_grad = True
    for param in model.pos_head.parameters():
        param.requires_grad = True
    summary(model, input_dims, device=device)

    loss_fn = nn.MSELoss()

    get_batches_start_time = time.time()
    test_dataloader = get_dataloader(params['test_sequences'], params, shuffle=True)
    get_batches_time = convert_time(time.time() - get_batches_start_time)
    print(f"Test dataloaders created in {get_batches_time}")

    if args.load_checkpoint:
        fpath = f'./checkpoints/{args.load_checkpoint}' + '.pt'
        print("Loading checkpoint from: \n" + fpath)
        checkpoint = torch.load(fpath, weights_only=True)
        model.load_state_dict(checkpoint)

        print("Evaluating checkpoint...")
        test_start_time = time.time()
        test_result = test(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
        test_time = convert_time(time.time() - test_start_time)
        print(f"Test results calculated in {test_time}")
        for recorder in test_result.values():
            recorder.save_results(folder_name=args.save_results, epoch=None)
    elif args.train:
        get_batches_start_time = time.time()
        train_dataloader = get_dataloader(params['train_sequences'], params, shuffle=True)
        get_batches_time = convert_time(time.time() - get_batches_start_time)
        print(f"Train dataloaders created in {get_batches_time}")
        train_and_eval(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            early_stopping_patience=params['early_stopping_patience'],
            best_val_loss=params['best_val_loss'],
            device=device,
            params=params,
            save_checkpoint=args.save_checkpoint,
            save_results=args.save_results
        )
    else:
        print("Neither train nor --load_checkpoint was specified! Exiting early...")
    

    if args.save_results:
        if args.train:
            generate_plots(os.path.join('./results/',args.save_results), "train", params["epochs"])
            generate_plots(os.path.join('./results/',args.save_results), "test", params["epochs"])
        elif args.load_checkpoint:
            generate_plots(os.path.join('./results/',args.save_results), "test", None)

    exec_time = convert_time(time.time() - exec_start_time)
    print(f"Execution completed in {exec_time}")




if __name__ == '__main__':
    main()
    print("Exiting...")