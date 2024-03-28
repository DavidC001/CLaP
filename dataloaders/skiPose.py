import os
import torch
from torch.utils.data import Dataset
import cv2
import random

import matplotlib.pyplot as plt
import torchvision.transforms as T


class ContrastiveSkiDataset(Dataset):
    def __init__(self, transform):

        # change this to the path where the dataset is stored
        self.data_path = "datasets/Ski-PosePTZ-CameraDataset-png"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['license.txt', 'load_h5_example.py', 'README.txt', 'load_h5_example.m']

        #train and test
        for dir in motion_seq:
          if dir not in no_dir:
            #seq_000 type of directory
            for seq in (os.listdir(os.path.join(self.data_path, dir))):
              if os.path.exists(os.path.join(self.data_path, dir, seq, 'cam_00')):
                        data_path = os.path.join(self.data_path, dir, seq, 'cam_00')
                        for lists in (os.listdir(data_path)):
                            paths.append(os.path.join(data_path, lists))

        self.data = {'paths': paths}


    def __len__(self):
        return len(self.data['paths'])

    def get_second_view(self, image_path):
        """Randomly gets another camera view"""
        split = image_path.split('/cam_00')
        random = random.randint(1, 5)
        split[2] = 'cam_0' + str(random)

        second_path = split[0] + '/cam_0' + str(random) + split[1]

        return second_path


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image1_path = self.data['paths'][idx]
        image2_path = self.get_second_view(image1_path)

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


class ClusterSkiDataset(Dataset):
    def __init__(self, transform, data_set='training'):
        print("ClusterSkiiDataset")

class PoseSkiDataset(Dataset):
    def __init__(self, transform, data_set='training'):
        print("PoseSkiiDataset")
