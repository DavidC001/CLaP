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
        self.data_path = dataset_dir+"/KinectDataset/side_"
        self.training_dir = []

        if mode == "train":
            self.data_path += "train_images"
        elif mode == "test":
            self.data_path += "test_images"
        

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)

        for dir in motion_seq:
            data_path = os.path.join(self.data_path, dir).replace('\\', '/')
            for lists in (os.listdir(data_path)):
                paths.append(os.path.join(data_path, lists).replace('\\', '/'))
            
        self.data = {'paths': paths}

        
    def __len__(self):
        return len(self.data['paths'])
    

    def __getitem__(self, idx):
        image_path = self.data['paths'][idx]
        top_view_path = image_path.replace("side", "top")

        image_side = cv2.imread(image_path)
        image_top = cv2.imread(top_view_path)

        image_side = cv2.cvtColor(image_side, cv2.COLOR_BGR2RGB)
        image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB)

        image_side = self.transform(image_side)
        image_top = self.transform(image_top)

        sample = dict()

        sample['image1'] = image_side
        sample['image2'] = image_top

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