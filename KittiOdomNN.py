import torch
import torch.nn as nn
from torch import Tensor
from ResidualBlock import ResidualBlock

# Used for validation
from torchsummary import summary
from device import get_device
from datasets.KittiOdomDataset import KittiOdomDataset
from torch.utils.data import DataLoader


class KittiOdomNN(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: tuple = (128, 128), device=None):
        super().__init__()

        # Initial convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=16, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # First block of residual layers and pooling
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Second block of residual layers and pooling
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Third block of residual layers and pooling
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Fourth block of residual layers and pooling
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.flatten = nn.Flatten(1)

        self.final = torch.nn.GRU(
            # input_size = int(image_size[0] / 2**4 * image_size[1] / 2**4 * 128),
            # TODO: for some reason the formula above doesn't match the size of the 4th layer.
            input_size = 226688,
            hidden_size = 8,
            device = device,
        )


        # Flattening and final linear layer
        # self.dropout = nn.Dropout(p=0.2)
        # self.linear = nn.Linear(
        #     in_features=int(image_size[0] / 2**4 * image_size[1] / 2**4 * 128),
        #     out_features=classes_num
        # )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(x)
        # x = self.dropout(x)
        # x = self.linear(x)
        x = self.final(x)

        return x
    
if __name__ == '__main__':
    device = get_device()
    model = KittiOdomNN(in_channels=3, image_size=(376, 1241), device=device)

    dataset = KittiOdomDataset(r'datasets\data_odometry_csv\00.csv')
    dataloader = DataLoader(dataset, batch_size=239, pin_memory=True)

    for batch_idx, sample in enumerate(dataloader):
        X = sample['image']
        out = model(X)
        print(type(out))
        break

    summary(model, (3, 376, 1241))

    # input = torch.randn(1, 3, 376, 1241).to(device=device)
    # out = model(input)
    # print(out)
