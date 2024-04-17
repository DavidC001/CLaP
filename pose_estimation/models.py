import sys
sys.path.append(".")

from torch import nn
import torch
from contrastive_training.simclr.model import get_simclr_net
from contrastive_training.simsiam.model import get_siam_net
from contrastive_training.MoCo.model import get_moco_net
from contrastive_training.supervised.model import get_supervised_net
from torchvision.models import resnet50

models = {
    'siam': get_siam_net,
    'simclr': get_simclr_net,
    'MoCo': get_moco_net,
    'LASCon': get_supervised_net,
    'ResNet50': resnet50
}


class Linear(nn.Module):
    def __init__(self, layers, output_dim=48):
        super(Linear, self).__init__()
        self.layers = nn.Sequential()
        layers.append(output_dim)
        for i in range(len(layers)-2):
            self.layers.add_module('linear'+str(i), nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-3:
                self.layers.add_module('relu'+str(i), nn.ReLU())


    def forward(self, x):
        z = self.layers(x)
        return z

def get_linear_evaluation_model(path, model_type, layers, out_dim, device='cuda'):
    base = models[model_type]()
    if path:
        base.load_state_dict(torch.load(path, map_location=torch.device(device)))

    if model_type == 'siam' or model_type == 'MoCo':
        base = base.base
    base.fc = Linear(layers, out_dim)


    return base