import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return x, self.layers(x)


def get_simclr_net(base_model='resnet18'):
    if base_model == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = MLP(2048, 2048, 128)
    elif base_model == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = MLP(512, 512, 128)
    else:
        raise ValueError("Invalid base model")
    model = nn.DataParallel(model)

    return model