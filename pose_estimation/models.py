import sys
sys.path.append(".")

from torch import nn
import torch
from contrastive_training.simclr.model import get_simclr_net
from contrastive_training.simsiam.model import get_siam_net
#from contrastive_training.MoCo.model import get_moco_net
#from contrastive_training.supervised.model import get_supervised_net
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

models = {
    'siam': get_siam_net,
    'simclr': get_simclr_net,
    #'MoCo': get_moco_net,
    'LASCon': get_simclr_net,
    'resnet': resnet50
}


class Linear(nn.Module):
    def __init__(self, layers, output_dim=48):
        super(Linear, self).__init__()
        self.layers = nn.Sequential()
        #attach 2048 to the beginning of the list
        layers.insert(0, 2048)
        layers.append(output_dim)
        for i in range(len(layers)-1):
            self.layers.add_module('linear'+str(i), nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-3:
                self.layers.add_module('relu'+str(i), nn.ReLU())


    def forward(self, x):
        z = self.layers(x)
        return z

def getPoseEstimModel(path, model_type, layers, out_dim, device='cpu'):
    if model_type != 'resnet':
        base = models[model_type]()
        base.load_state_dict(torch.load(path, map_location=torch.device(device))).to(device)
    elif model_type == 'resnet':
        weights = ResNet50_Weights.DEFAULT
        base = nn.DataParallel(resnet50(weights=weights)).to(device)

    if model_type == 'siam' or model_type == 'MoCo':
        base.module = base.module.base
    base.module.fc = Linear(layers, out_dim)

    return base.to(device)