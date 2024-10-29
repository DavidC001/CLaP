import os
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import cv2
import re
import random
import torchvision.transforms as transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
import torchvision.transforms as T

class Projector(nn.Module):
    def __init__(self, input_dim, out_dim, hidder_proj):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidder_proj),
            nn.BatchNorm1d(hidder_proj),
            nn.ReLU(inplace=True),

            nn.Linear(hidder_proj, hidder_proj),
            nn.BatchNorm1d(hidder_proj),
            nn.ReLU(inplace=True),

            nn.Linear(hidder_proj, out_dim),
        )

    def forward(self, x):
        return x, self.proj(x)

class SiamMLP(nn.Module):
    def __init__(self, base, input_dim, out_dim, hidder_proj, hidden_pred):
        super().__init__()

        self.base = base

        base.fc = Projector(input_dim, out_dim, hidder_proj)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_pred),
            nn.BatchNorm1d(hidden_pred),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_pred, out_dim)
        )


    def forward(self, x):
        x, projections =  self.base(x)

        predictions = self.predictor(projections)

        return x, projections.detach(), predictions

from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights


def get_siam_net(base_model='resnet18'):
    """
    Get the SimSiam model with the specified base model

    Parameters:
        base_model (str): Base model to use for the SimSiam model

    Returns:
        nn.Module: SimSiam model
    """
    if base_model == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model = SiamMLP(model, 2048, 2048, 2048, 512)
    elif base_model == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model = SiamMLP(model, 512, 512, 512, 128)
    else:
        raise ValueError("Invalid base model")
    model = nn.DataParallel(model)

    return model

