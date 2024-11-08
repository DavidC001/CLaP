import sys

sys.path.append(".")

from torch import nn
import torch
from contrastive_training.simclr.model import get_simclr_net
from contrastive_training.simsiam.model import get_siam_net
from contrastive_training.MoCo.model import get_moco_net

# from contrastive_training.supervised.model import get_supervised_net
from torchvision.models import resnet50, resnet18
from torchvision.models import ResNet50_Weights, ResNet18_Weights


models = {
    "simsiam": get_siam_net,
    "simclr": get_simclr_net,
    'MoCo': get_moco_net,
    "LASCon": get_simclr_net,
    "resnet": None,
}


class Estimator(nn.Module):
    """
        Estimation layer for pose estimation.
    """

    def __init__(self, layers, output_dim=48, base_model="resnet18", layer_norm=True, activation="gelu"):
        """
        Initialize the Linear layer.

        Parameters:
            layers: list, list of layer dimensions
            output_dim: int, output dimension, default is 48
            base_model: str, base model, default is 'resnet18'
            layer_norm: bool, whether to use layer normalization, default is True
            activation: str, activation function, ['gelu', 'relu'], default is 'gelu'
        """
        super(Estimator, self).__init__()
        self.layers = nn.Sequential()
        # attach 2048 to the beginning of the list
        if base_model == "resnet50":
            layers.insert(0, 2048)
        elif base_model == "resnet18":
            layers.insert(0, 512)
        else:
            raise ValueError("Invalid base model")
        layers.append(output_dim)
        if layer_norm:
            self.LN = nn.LayerNorm(layers[0])
        for i in range(len(layers) - 1):
            self.layers.add_module(
                "linear" + str(i), nn.Linear(layers[i], layers[i + 1])
            )
            if i < len(layers) - 2:
                self.layers.add_module("gelu" + str(i), nn.GELU() if activation == "gelu" else nn.ReLU())

    def forward(self, x):
        if hasattr(self, "LN"):
            x = self.LN(x)
        z = self.layers(x)
        return z


def getPoseEstimModel(
    path,
    model_type,
    layers,
    out_dim,
    device="cpu",
    base_model="resnet18",
    layer_norm=True,
    activation="gelu",
):
    """
    Get the pose estimation model.

    Parameters:
        path: str, path to the model weights
        model_type: str, model type ['simsiam', 'simclr', 'MoCo', 'LASCon', 'resnet']
        layers: list, list of layer dimensions
        out_dim: int, output dimension
        device: str, device, default is 'cpu'
        base_model: str, base model, default is 'resnet18'
        layer_norm: bool, whether to use layer normalization, default is True
        activation: str, activation function, default is 'gelu'

    Returns:
        base: torch.nn.Module, pose estimation model with the specified weights
    """
    if model_type != "resnet":
        base = models[model_type](base_model)
        base.load_state_dict(torch.load(path, map_location=torch.device(device)))
    else:
        if base_model == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            base = nn.DataParallel(resnet50(weights=weights))
        elif base_model == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            base = nn.DataParallel(resnet18(weights=weights))
        else:
            raise ValueError("Invalid base model")

    base = base.to(device)

    if model_type == "simsiam":
        base.module = base.module.base
    elif model_type == "moco":
        base.module = base.module.encoder_q
    
    base.module.fc = Estimator(layers, out_dim, base_model, layer_norm, activation)

    return base.to(device)
