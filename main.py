import argparse
import torch
from torch import nn
from torchsummary import summary

# Local imports
from utils.KittiOdomNN import KittiOdomNN
from utils.KittiOdomDataset import get_dataloaders
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
    parser.add_argument('--save_train_results', type=str, default=None, help='Saves train results to .csv file. If ommitted, will not save results.')
    parser.add_argument('--test', action='store_true', help='Add arg to evaluate model. Otherwise skips evaluation step.')
    parser.add_argument('--save_test_results', type=str, default=None, help='Saves eval results to .csv file. If ommitted, will not save results.')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Local path to .pt which contains model checkpoint')
    parser.add_argument('--save_checkpoint', type=str, default=None, help='Local directory to save model checkpoints. If ommitted, will not save checkpoint.')
    parser.add_argument('--params', type=str, default=None, help='Yaml filename under ./checkpoints which contains hyperparameters to tune. If ommitted, will use default params from default.yaml')

    args = parser.parse_args()
    assert not (args.train and args.load_checkpoint), "Selecting both training and --load_checkpoint is not supported"
    if args.save_train_results:
        assert args.train, "Training is not selected but saving train results is."
    return args


def main():
    args = get_argparser()
    params = load_params(args.params)

    device = get_device()
    model = KittiOdomNN(gru_hidden_size=params['gru_hidden_size'], in_channels=3, device=device).to(device)
    # TODO: loss fn and image size should be set outside of this function
    loss_fn = nn.MSELoss()
    summary(model, (3, 376, 1241), device=device)


    if args.test:
        test_dataloaders = get_dataloaders(params['test_sequences'], params['batch_size'])
    else:
        test_dataloaders = []

    if args.load_checkpoint:
        fpath = f'./checkpoints/{args.load_checkpoint}' + '.pt'
        print("Loading checkpoint from: \n" + fpath)
        checkpoint = torch.load(fpath, weights_only=True)
        model.load_state_dict(checkpoint)
        if args.test:
            print("Evaluating checkpoint...")
            test_results = test(dataloaders=test_dataloaders, model=model, loss_fn=loss_fn, device=device)
    elif args.train:
        train_dataloaders = get_dataloaders(params['train_sequences'], params['batch_size'])
        train_results, test_results = train_and_eval(
            train_dataloaders=train_dataloaders,
            test_dataloaders=test_dataloaders,
            model=model,
            loss_fn=loss_fn,
            device=device,
            params=params,
            run_test=args.test,
        )
    else:
        print("Neither train nor --load_checkpoint was specified! Exiting early...")
        return
    
    if args.save_checkpoint:
        fpath = f'./checkpoints/{args.save_checkpoint}' + '.pt'
        print("Saving model checkpoint to: \n" + fpath + '.pt')
        torch.save(model.state_dict(), fpath)

    if args.save_train_results:
        fpath = f'./results/train/'
        print("Saving train results to: \n" + fpath + f'\n with name {args.save_train_results}')
        for epoch, result in enumerate(train_results):
            result.to_csv(fpath + args.save_train_results + f'_epoch_{epoch}.csv')

    if args.save_test_results:
        if args.load_checkpoint: # Only 1 recorder
            fpath = f'./results/test/{args.save_test_results}' + '.csv'
            print("Saving test results to: \n" + fpath + '.csv')
            test_results.to_csv(fpath)
        else: # Otherwise multiple recorders, per epoch
            fpath = f'./results/test/'
            print("Saving test results to: \n" + fpath + f'\n with name {args.save_test_results}')
            for epoch, result in enumerate(test_results):
                result.to_csv(fpath + args.save_test_results + f'_epoch_{epoch}.csv')





if __name__ == '__main__':
    main()
    print("Exiting...")