import os
import torch
from torch.utils.data import Dataset
import cv2
import random

import matplotlib.pyplot as plt
import torchvision.transforms as T


class SkiiDataset(Dataset):
    def __init__(self, transform):
        print("SkiiDataset")

class ClusterSkiiDataset(Dataset):
    def __init__(self, transform, data_set='training'):
        print("ClusterSkiiDataset")