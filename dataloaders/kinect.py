import os
import torch
from torch.utils.data import Dataset
import cv2
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T


class ContrastiveKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):
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


def getContrastiveDatasetKinect(transform, dataset_dir="datasets"):
    """
    Returns a tuple of train and test datasets for contrastive learning using Kinect data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    train = ContrastiveKinectDataset(transform, dataset_dir, mode="train")
    test = ContrastiveKinectDataset(transform, dataset_dir, mode="test")
    return train, test



class ClusterKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):
        print("ClusterKinectDataset")

        self.data_path = dataset_dir+"/ClusterKinectDataset"
        
class PoseKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):
        print("PoseKinectDataset")

def getPoseDatasetKinect(transform, dataset_dir="datasets"):
    """
    Returns a tuple of train and test datasets for pose estimation using Kinect data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    train = PoseKinectDataset(transform, dataset_dir, mode="train")
    test = PoseKinectDataset(transform, dataset_dir, mode="test")
    return train, test