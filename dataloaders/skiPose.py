import os
import torch
from torch.utils.data import Dataset
import cv2
import random

import matplotlib.pyplot as plt
import torchvision.transforms as T
import math

#PIL image
from PIL import Image

import h5py

generator = torch.Generator().manual_seed(42)


class ContrastiveSkiDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"
        self.training_dir = []

        self.transform = transform

        paths = []

        #train or test
        if mode == 'train':
          dir = 'train'
        else:
          dir = 'test'

        for seq in (os.listdir(os.path.join(self.data_path, dir).replace('\\', '/'))):
          if os.path.exists(os.path.join(self.data_path, dir, seq, 'cam_00').replace('\\', '/')):
            image_path = os.path.join(self.data_path, dir, seq, 'cam_00').replace('\\', '/')
            for lists in (os.listdir(image_path)):
                paths.append(os.path.join(image_path, lists).replace('\\', '/'))

        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def get_second_view(self, image_path):
        """Randomly gets another camera view"""
        split = image_path.split('/cam_0')
        rand = random.randint(0, 5)

        second_path = split[0] + '/cam_0' + str(rand) + split[1][1:]

        return second_path


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image1_path = self.data['paths'][idx]

        for i in range(10):
            new_image_path = self.get_second_view(image1_path)
            if os.path.exists(new_image_path):
                image1_path = new_image_path
                break
        image1 = cv2.imread(image1_path)
        image1 =cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image2_path = None
        for i in range(10):
            new_image_path = self.get_second_view(image1_path)
            if os.path.exists(new_image_path) and new_image_path != image1_path:
                image2_path = new_image_path
                break

        if image2_path is None:
            #random rotation
            image2 = T.RandomRotation(45)(Image.fromarray(image1))
        else:
            image2 = cv2.imread(image2_path)
            image2 =cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        image2 = self.transform(image2)
        image1 = self.transform(image1)

        sample['image1'] = image1
        sample['image2'] = image2

        return sample

class CompleteContrastiveSkiDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train", drop=0):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"
        self.training_dir = []

        self.transform = transform

        poses = {}
        cameras = ["cam_00", "cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]

        #train or test
        if mode == 'train':
          dir = 'train'
        else:
          dir = 'test'

        for seq in (os.listdir(os.path.join(self.data_path, dir).replace('\\', '/'))):
            #if seq is a directory
            if os.path.exists(os.path.join(self.data_path, dir, seq, 'cam_00').replace('\\', '/')):
                poses[seq] = {}
                for cam in cameras:
                    if os.path.exists(os.path.join(self.data_path, dir, seq, cam).replace('\\', '/')):
                        image_path = os.path.join(self.data_path, dir, seq, cam).replace('\\', '/')
                        for image in (os.listdir(image_path)):
                            if image not in poses[seq]:
                                poses[seq][image] = []
                            poses[seq][image].append(os.path.join(image_path, image).replace('\\', '/'))

        #generate all possible pairs inside the same sequence with different cameras
        paths = []
        for seq in poses:
            for image in poses[seq]:
                if len(poses[seq][image]) > 1:
                    paths_pose = []
                    for i in range(len(poses[seq][image])):
                        for j in range(i+1, len(poses[seq][image])):
                            paths_pose.append((poses[seq][image][i], poses[seq][image][j]))
                    #drop some pairs
                    paths.extend(random.sample(paths_pose, math.ceil(len(paths_pose) * (1-drop))))
                else:
                    paths.append((poses[seq][image][0], None))

        # breakpoint()
        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image1_path = self.data['paths'][idx][0]
        image1 = cv2.imread(image1_path)
        image1 =cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image2_path = self.data['paths'][idx][1]
        if image2_path is None:
            #random rotation
            image2 = T.RandomRotation(45)(Image.fromarray(image1))
        else:
            image2 = cv2.imread(image2_path)
            image2 =cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        image2 = self.transform(image2)
        image1 = self.transform(image1)

        sample['image1'] = image1
        sample['image2'] = image2

        return sample


def getContrastiveDatasetSki(transform, dataset_dir="datasets", use_complete=True, drop=0.5):
    """
    Returns a tuple of train and test datasets for contrastive learning using Ski data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".
        use_complete (bool, optional): Whether to use complete pairs. Defaults to True.
        drop (float, optional): The percentage of pairs to drop. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    if use_complete:
        dataset = CompleteContrastiveSkiDataset(transform, dataset_dir, mode="train", drop=drop)
        test = CompleteContrastiveSkiDataset(transform, dataset_dir, mode="test", drop=drop)
    else:
        dataset = ContrastiveSkiDataset(transform, dataset_dir, mode="train")
        test = ContrastiveSkiDataset(transform, dataset_dir, mode="test")

    num_samples = len(dataset)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    train, val = torch.utils.data.random_split(
        dataset, [training_samples, val_samples], generator=generator
    )

    return train, val, test


class ClusterSkiDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", set="train", use_cluster="NONE"):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"

        paths = []

        

        motion_seq = os.listdir(self.data_path)
        if set == 'train':
            no_dir = ['license.txt', 'load_h5_example.py', 'README.txt', 'load_h5_example.m', 'test']
        else:
            no_dir = ['license.txt', 'load_h5_example.py', 'README.txt', 'load_h5_example.m', 'train']

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

        sample['path'] = image_path
        sample['image'] = image

        return sample

class PoseSkiDataset(Dataset):

    def __init__(self, transform, dataset_dir="datasets", mode = "train", use_cluster="NONE"):

        # change this to the path where the dataset is stored
        data_path = dataset_dir+"/Ski-PosePTZ-CameraDataset-png"
        self.training_dir = []

        self.transform = transform

        paths = []

        #train or test
        if mode == 'train':
            dir = '/train'
        else:
            dir = '/test'

        path_file = data_path+dir+'/labels.h5'
        h5_label_file = h5py.File(path_file, 'r')
        included_images = []
        
        if not use_cluster.startswith("RANDOM") and use_cluster != "NONE":
            with open(use_cluster, 'r') as f:
                included_images = f.readlines()
        
        #load image's path in order
        for index in range(0,len(h5_label_file['cam'])):
            seq   = int(h5_label_file['seq'][index])
            cam   = int(h5_label_file['cam'][index])
            frame = int(h5_label_file['frame'][index])
            image_path = data_path+dir+'/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq,cam,frame)
            if len(included_images) == 0 or image_path in included_images:
                paths.append(image_path.replace('\\','/'))

        if use_cluster.startswith("RANDOM"):
            percent = int(use_cluster.split("_")[-1])
            num_samples = len(paths)
            paths = random.sample(paths, math.ceil(num_samples * percent / 100))
            

        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        #read the image
        image = cv2.imread(self.data['paths'][idx])
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        sample['image'] = image
        
        #load the joints position
        path_file = self.data['paths'][idx].split('/seq')[0]+'/labels.h5'
        h5_label_file = h5py.File(path_file, 'r')
        poses_3d = (h5_label_file['3D'][idx])

        sample['poses_3d'] =  poses_3d

        #camera param
        intrinsic = h5_label_file['cam_intrinsic'][idx].reshape([-1,3])
        traslation = h5_label_file['cam_position'][idx]
        rotation = h5_label_file ['R_cam_2_world'][idx].reshape([3,3])
        cam = {'K':intrinsic, 'R':rotation, 't':traslation}

        sample['cam'] = cam

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
    
    num_samples = len(train)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    train, val = torch.utils.data.random_split(
        train, [training_samples, val_samples], generator=generator
    )

    test = PoseSkiDataset(transform, dataset_dir, mode="test")
    
    return train, val, test
