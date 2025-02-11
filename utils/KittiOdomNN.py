import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


class KittiOdomNN(nn.Module):
    def __init__(self, in_channels, gru_hidden_size, device=None):
        super().__init__()

        # Initial convolution layer
        # conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7,7), stride=2, padding=3)
        # conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=2, padding=2)
        # conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        # maxPool = nn.MaxPool2d(kernel_size=(3,3))
        # self.conv = nn.Sequential(
        #     conv1,
        #     nn.ReLU(),
        #     maxPool,
        #     conv2,
        #     nn.ReLU(),
        #     maxPool,
        #     conv3,
        #     nn.ReLU(),
        #     maxPool,
        # )

        resnet50 = torchvision.models.resnet50(pretrained=True)

        # Adjust first layer to correspond to in_channels=6
        if in_channels == 6:
            conv1_weights = resnet50.conv1.weight.data
            new_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv1.weight.data[:, :3, :, :] = conv1_weights
            new_conv1.weight.data[:, 3:, :, :] = conv1_weights

            resnet50.conv1 = new_conv1
        
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1]) # Remove FC layer
        
        self.flatten = nn.Flatten(1)

        self.dropout = nn.Dropout(p=0.2)

        # self.gru = [nn.GRU(
        #     input_size = 28336,
        #     hidden_size = gru_hidden_size,
        #     device = device,
        # ) for _ in range(12)]

        self.gru = nn.GRU(
            input_size = 2048,
            hidden_size = gru_hidden_size,
            num_layers=2,
            batch_first=True,
            device=device
        )

        # self.gru = nn.GRU(
        #     input_size = 453376,
        #     hidden_size = gru_hidden_size,
        #     num_layers=2,
        #     device=device
        # )

        self.rot_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 16, device=device),
            nn.ReLU(),
            nn.Linear(16, 9, device=device),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 16, device=device),
            nn.ReLU(),
            nn.Linear(16, 3, device=device),
        )


    def forward(self, x: Tensor) -> Tensor:
        # x = self.conv(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.dropout(x)

        # final_outs = []
        # for module in self.gru:
        #     output = module(x)
        #     final_outs.append(output[0][:,-1])
        # return torch.stack(final_outs, dim=1)

        x, _ = self.gru(x)
        # return self.fc(x)

        return self.rot_head(x), self.pos_head(x)

if __name__ == '__main__':
    from DeviceLoader import get_device
    device = get_device()

    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # print(model)
    # summary(model, (3, 376, 1241), device=device)
    model = KittiOdomNN(in_channels=6, gru_hidden_size=16, device=device)
    summary(model, (6, 376, 1241), device=device)
