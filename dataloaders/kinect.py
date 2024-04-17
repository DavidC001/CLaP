import os
import torch
from torch.utils.data import Dataset
import cv2
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
import h5py 


class ContrastiveKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):
        #print("KinectDataset")
        
        data = h5py.File(dataset_dir+"/ITOP/ITOP_side_" + mode + "_images.h5", 'r')

        print(data.keys())
        
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

        self.data_path = dataset_dir+"/KinectDataset/"
        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)

        for dir in motion_seq:
            data_path = os.path.join(self.data_path, dir).replace('\\', '/')
            for lists in (os.listdir(data_path)):
                if lists.endswith('.jpg'):
                    paths.append(os.path.join(data_path, lists).replace('\\', '/'))

        #randoml select up to 6000 images
        paths = random.sample(paths, min(6000, len(paths)))
        
        self.data = {'paths': paths}

        
    def __len__(self):
        return len(self.data['paths'])
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image_path = self.data['paths'][idx]

        image = cv2.imread(image_path)
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        sample['image'] = image

        return sample


class PoseKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):
        print("PoseKinectDataset")

        self.data_path = dataset_dir+"/KinectDataset/"
        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)

        path_file = dataset_dir + "/KinectDataset/labels.h5"
        h5_label_file = h5py.File(path_file, 'r')

        for dir in range(0, len(h5_label_file['seq'])):
            seq = int(h5_label_file['seq'][dir])
            cam = int(h5_label_file['cam'][dir])
            frame = int(h5_label_file['frame'][dir])
            subj = int(h5_label_file['subj'][dir])
            pose_3D = h5_label_file['3D'][dir].reshape([-1,3])
            pose_2D = h5_label_file['2D'][dir].reshape([-1,2])
        self.data = {'paths': paths}

        
    

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


if __name__ == "__main__":
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    dataset = ContrastiveKinectDataset(transform)
    