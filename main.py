import argparse
import torch
from torch import nn
from torchsummary import summary

# Force utf-8 encoding to resolve issues printing to terminal
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# Local imports
from utils.KittiOdomNN import KittiOdomNN
from utils.KittiOdomDataset import get_batches
from utils.DeviceLoader import get_device
from utils.ParamLoader import load_params
from utils.Trainer import train_and_eval
from utils.Tester import test

def get_argparser():
    # Parse input arguments
    parser = argparse.ArgumentParser(
        prog='Kitti Odometery NN',
        description='Training & Evaluator for the Kitti Odometery dataset'
    )
    parser.add_argument('--train', action='store_true', help='Add arg to train. Otherwise uses .pt specified in --checkpoint')
    parser.add_argument('--save_results', type=str, default=None, help='Saves results to .csv files under a subdirectory. If ommitted, will not save results.')
    parser.add_argument('--test', action='store_true', help='Add arg to evaluate model. Otherwise skips evaluation step.')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Local path to .pt which contains model checkpoint')
    parser.add_argument('--save_checkpoint', type=str, default=None, help='Local directory to save model checkpoints. If ommitted, will not save checkpoint.')
    parser.add_argument('--params', type=str, default=None, help='Yaml filename under ./checkpoints which contains hyperparameters to tune. If ommitted, will use default params from default.yaml')

    args = parser.parse_args()
    assert not (args.train and args.load_checkpoint), "Selecting both training and --load_checkpoint is not supported"
    return args


def main():
    args = get_argparser()
    params = load_params(args.params)

    device = get_device()
    input_dims = [params['img_size'] for _ in range(2)] if params['stack_input_images'] else params['img_size']
    model = KittiOdomNN(stack_input=params['stack_input_images'], gru_hidden_size=params['gru_hidden_size'], device=device).to(device)
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

    if args.test:
        test_dataloaders = get_batches(params['test_sequences'], params)
    else:
        test_dataloaders = []

    if args.load_checkpoint:
        fpath = f'./checkpoints/{args.load_checkpoint}' + '.pt'
        print("Loading checkpoint from: \n" + fpath)
        checkpoint = torch.load(fpath, weights_only=True)
        model.load_state_dict(checkpoint)
        if args.test:
            print("Evaluating checkpoint...")
            test_result = test(dataloaders=test_dataloaders, model=model, loss_fn=loss_fn, device=device)
            test_result.save_results(folder_name=args.save_results, epoch=None)
    elif args.train:
        train_dataloaders = get_batches(params['train_sequences'], params)
        train_and_eval(
            train_dataloaders=train_dataloaders,
            test_dataloaders=test_dataloaders,
            model=model,
            loss_fn=loss_fn,
            device=device,
            params=params,
            run_test=args.test,
            save_checkpoint=args.save_checkpoint,
            save_results=args.save_results
        )
    else:
        print("Neither train nor --load_checkpoint was specified! Exiting early...")
        return
    
    # if args.save_checkpoint:
    #     if not os.path.exists('./checkpoints'):
    #         os.makedirs('./checkpoints')
    #     fpath = f'./checkpoints/{args.save_checkpoint}' + '.pt'
    #     print("Saving model checkpoint to: \n" + fpath)
    #     torch.save(model.state_dict(), fpath)

    # if args.save_results:
    #     print(f"Saving results to ./results/{args.save_results}")
    #     save_results(args.save_results, train_results, test_results)





if __name__ == '__main__':
    main()
    print("Exiting...")