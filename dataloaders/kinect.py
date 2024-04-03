import os
import torch
from torch.utils.data import Dataset
import cv2
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T


class ContrastiveKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):
        print("KinectDataset")
        self.data_path = dataset_dir+"/KinectDataset/"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = []

        
    def __len__(self):
        return len(self.data['paths'])
    

    def __getitem__(self, idx):
        
        return sample


class ClusterKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):
        print("ClusterKinectDataset")

        self.data_path = dataset_dir+"/ClusterKinectDataset"
        


class PoseKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):
        print("PoseKinectDataset")