import os
import torch
from torch.utils.data import Dataset
import cv2
import random

import matplotlib.pyplot as plt
import torchvision.transforms as T

import h5py
import imageio

generator = torch.Generator().manual_seed(42)


class ContrastiveSkiDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['license.txt', 'load_h5_example.py', 'README.txt', 'load_h5_example.m']

        #train or test
        if mode == 'train':
          dir = '/train'
        else:
          dir = '/test'

        for seq in (os.listdir(os.path.join(self.data_path, dir).replace('\\', '/'))):
          if os.path.exists(os.path.join(self.data_path, dir, seq, 'cam_00').replace('\\', '/')):
            data_path = os.path.join(self.data_path, dir, seq, 'cam_00').replace('\\', '/')
          for lists in (os.listdir(data_path)):
            paths.append(os.path.join(data_path, lists).replace('\\', '/'))

        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def get_second_view(self, image_path):
        """Randomly gets another camera view"""
        split = image_path.split('/cam_00')
        random = random.randint(0, 5)

        second_path = split[0] + '/cam_0' + str(random) + split[1]

        return second_path


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image1_path = self.data['paths'][idx]
        image2_path = self.get_second_view(image1_path)

        #make the first image random
        while True:
          image1_path = self.get_second_view(image1_path)
          if image1_path != image2_path:
            break


        for i in range(0, 10):
            if os.path.isfile(image2_path):
                image2 = cv2.imread(image2_path)
                image2 =cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = self.transform(image2)
                break
            else:
                image2_path = self.get_second_view(image1_path)
        else:
            # apply random rotation on the first image if the second view cannot be found in 10 iterations
            image2 = cv2.imread(image1_path)
            image2 =cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = self.transform(image2)
            image2 = T.RandomRotation(45)(image2)

        image1 = cv2.imread(image1_path)
        image1 =cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = self.transform(image1)

        sample['image1'] = image1
        sample['image2'] = image2

        return sample


def getContrastiveDatasetSki(transform, dataset_dir="datasets"):
    """
    Returns a tuple of train and test datasets for contrastive learning using Ski data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    dataset = ContrastiveSkiDataset(transform, dataset_dir, mode="train")
    test = ContrastiveSkiDataset(transform, dataset_dir, mode="test")

    num_samples = len(dataset)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    train, val, = torch.utils.data.random_split(
        dataset, [training_samples, val_samples], generator=generator
    )

    return train, val, test


class ClusterSkiDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['license.txt', 'load_h5_example.py', 'README.txt', 'load_h5_example.m']

        #train and test
        for dir in motion_seq:
          if dir not in no_dir:
            #seq_000 type of directory
            for seq in (os.listdir(os.path.join(self.data_path, dir).replace('\\', '/'))):
                #if seq is a directory
                if os.path.exists(os.path.join(self.data_path, dir, seq, 'cam_00').replace('\\', '/')):
                    for cams in range(6):
                        data_path = os.path.join(self.data_path, dir, seq, 'cam_0'+str(cams)).replace('\\', '/')
                        for lists in (os.listdir(data_path)):
                            paths.append(os.path.join(data_path, lists).replace('\\', '/'))


        self.transform = transform

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

class PoseSkiDataset(Dataset):

    def __init__(self, transform, dataset_dir="datasets", mode = "train"):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['license.txt', 'load_h5_example.py', 'README.txt', 'load_h5_example.m']

        #train or test
        if mode == 'train':
          dir = '/train'
        else:
          dir = '/test'

        path_file = '/content/Ski-PosePTZ-CameraDataset-png'+dir+'/labels.h5'
        h5_label_file = h5py.File(path_file, 'r')

        #load image's path in order
        for index in range(0,len(h5_label_file['cam'])):
          seq   = int(h5_label_file['seq'][index])
          cam   = int(h5_label_file['cam'][index])
          frame = int(h5_label_file['frame'][index])
          image_path = dataset_dir+dir+'/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq,cam,frame)
          paths.append(image_path.replace('\\','/'))

        self.data = {'paths': paths, 'mode':mode}

    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        #read the image
        image = imageio.imread(self.data['paths'][idx])

        sample['image'] = image

        dir = self.data['mode']
        
        #load the joints position
        path_file = '/content/Ski-PosePTZ-CameraDataset-png'+dir+'/labels.h5'
        h5_label_file = h5py.File(path_file, 'r')
        poses_3d = (h5_label_file['3D'][idx].reshape([-1,3]))

        sample['poses_3d'] =  poses_3d

        #camera param
        #TO DO

        #sample['cam'] = cam

        return sample

def getPoseDatasetSki(transform, dataset_dir="datasets"):
    """
    Returns a tuple of train and test datasets for pose estimation using Ski data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    train = PoseSkiDataset(transform, dataset_dir, mode="train")
    test = PoseSkiDataset(transform, dataset_dir, mode="test")
    return train, test
