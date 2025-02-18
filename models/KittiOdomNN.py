import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torchsummary import summary


class KittiOdomNN(nn.Module):
    def __init__(self, gru_hidden_size: int, device=None):
        super().__init__()

        backbone_pretrained = torchvision.models.efficientnet_b0(pretrained=True)
        
        self.backbone = nn.Sequential(*list(backbone_pretrained.children())[:-1]) # Remove FC layer
        
        self.flatten = nn.Flatten(1)

        self.dropout = nn.Dropout(p=0.2)

        self.gru = nn.GRU(
            input_size = 2560,
            hidden_size = gru_hidden_size,
            num_layers=2,
            batch_first=True,
            device=device
        )

        self.rot_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 16, device=device),
            nn.ReLU(),
            nn.Linear(16, 3, device=device),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 16, device=device),
            nn.ReLU(),
            nn.Linear(16, 3, device=device),
        )


    def forward(self, x_prev: Tensor, x_curr: Tensor) -> Tensor:
        x_curr = self.backbone(x_curr)
        x_curr = self.flatten(x_curr)
        x_curr = self.dropout(x_curr)

        x_prev = self.backbone(x_prev)
        x_prev = self.flatten(x_prev)
        x_prev = self.dropout(x_prev)

        x = torch.cat((x_curr, x_prev), dim=1)

        x, _ = self.gru(x)

        return self.rot_head(x), self.pos_head(x)

if __name__ == '__main__':
    from DeviceLoader import get_device
    device = get_device()

    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    # print(model)
    # summary(model, (3, 376, 1241), device=device)
    model = KittiOdomNN(gru_hidden_size=1024, device=device)
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
    # print(model)
    summary(model, [(3, 376, 1241), (3, 376, 1241)], device=device)
