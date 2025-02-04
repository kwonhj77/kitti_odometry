import torch
import torch.nn as nn
from torch import Tensor
import time
from trainer import train
from tester import test

# Used for validation
from torchsummary import summary
from device import get_device
from datasets.KittiOdomDataset import KittiOdomDataset
from torch.utils.data import DataLoader, ConcatDataset

# image_size=(376, 1241)
NUM_EPOCHS = 3
BATCH_SIZE = 19
class KittiOdomNN(nn.Module):
    def __init__(self, in_channels: int = 3, device=None):
        super().__init__()

        # Initial convolution layer
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=1, padding=0)
        conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, padding=0)
        conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        maxPool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv = nn.Sequential(
            conv1,
            nn.ReLU(),
            maxPool,
            conv2,
            nn.ReLU(),
            maxPool,
            conv3,
            nn.ReLU(),
            maxPool,
        )
        
        self.flatten = nn.Flatten(1)

        self.dropout = nn.Dropout(p=0.2)

        self.final = [torch.nn.GRU(
            input_size = 28336,
            hidden_size = 8,
            device = device,
        ) for _ in range(12)]

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)

        final_outs = []
        for module in self.final:
            output = module(x)
            final_outs.append(output[0][:,-1])
        return torch.stack(final_outs, dim=1)

    
if __name__ == '__main__':
    device = get_device()
    model = KittiOdomNN(in_channels=3, device=device).to(device)

    all_datasets = []
    for sequence in range(0,11):
        if sequence < 10:
            sequence = f"0{sequence}"
        else:
            sequence = str(sequence)
        fpath = f'datasets/data_odometry_csv/{sequence}.csv'
        all_datasets.append(KittiOdomDataset(fpath))

    train_dataset = ConcatDataset(dataset for dataset in all_datasets[:7])
    test_dataset = ConcatDataset(dataset for dataset in all_datasets[7:])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    # for batch_idx, sample in enumerate(dataloader):
    #     X = sample['image']
    #     out = model(X.to(device))
    #     print(type(out))
    #     break

    summary(model, (3, 376, 1241), device=device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08)

    print("Training start")
    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        epoch_start = time.time()
        train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
        epoch_end = time.time() - epoch_start
        print(f"Epoch {epoch+1} Time Elapsed: {epoch_end}")

    train_end = time.time() - train_start
    print(f"Time Elapsed during training: {train_end}")



