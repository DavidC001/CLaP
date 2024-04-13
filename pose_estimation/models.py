from torch import nn
import torch

class Linear(nn.Module):
    def __init__(self, layers):
        super(Linear, self).__init__()
        self.layers = nn.Sequential()
        for layer in enumerate(layers):
            self.layers.add_module(f'linear_{layer[0]}', nn.Linear(layer[0], layer[1]))
            if layer[0] != len(layers) - 1:
                self.layers.add_module(f'relu_{layer[0]}', nn.ReLU())


    def forward(self, x):
        z = self.layers(x)
        return z

def get_linear_evaluation_model(path, base, model_type, device='cuda'):

    base.load_state_dict(torch.load(path, map_location=torch.device(device)))

    if model_type == 'siam':
        base = base.base
        base.fc = Linear()
    else:
        base.fc = Linear()

    return base