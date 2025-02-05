import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


# TODO: Move this to a config files
# image_size=(376, 1241)
# TRAIN_SEQUENCES = (0,7)
# TEST_SEQUENCES = (7,11)
# NUM_EPOCHS = 1
# BATCH_SIZE = 128
# LR = 0.00001
# GRU_HIDDEN_SIZE = 8


class KittiOdomNN(nn.Module):
    def __init__(self, in_channels, gru_hidden_size, device=None):
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

        # self.gru = [torch.nn.GRU(
        #     input_size = 28336,
        #     hidden_size = gru_hidden_size,
        #     device = device,
        # ) for _ in range(12)]

        self.gru = torch.nn.GRU(
            input_size = 28336,
            hidden_size = gru_hidden_size,
            device=device
        )

        self.fc = torch.nn.Linear(8, 12, device=device)


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)

        # final_outs = []
        # for module in self.gru:
        #     output = module(x)
        #     final_outs.append(output[0][:,-1])
        # return torch.stack(final_outs, dim=1)

        x, _ = self.gru(x)
        return self.fc(x)

if __name__ == '__main__':
    from DeviceLoader import get_device
    device = get_device()
    model = KittiOdomNN(gru_hidden_size=8, in_channels=3, device=device)
    summary(model, (3, 376, 1241), device=device)
