import os
import torch
from torch.utils.data import Dataset
import cv2
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision.transforms as T

import json
import numpy as np
import math


generator = torch.Generator()


class ContrastivePanopticDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):

        #open green_images.txt
        self.no_files = []
        with open(dataset_dir+"/green_images.txt") as f:
            for line in f:
                self.no_files.append(line.strip())

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/ProcessedPanopticDataset/"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['scripts','python','matlab','.git','glViewer.py','README.md','matlab',
                'README_kinoptic.md', '171204_pose3']

        for dir in motion_seq:
            if dir not in no_dir:
                if 'haggling' in dir:
                    continue
                elif dir == '171204_pose2' or dir =='171204_pose5' or dir =='171026_cello3':
                    if os.path.exists(os.path.join(self.data_path,dir, 'hdJoints')):
                        data_path = os.path.join(self.data_path,dir, 'hdJoints')
                        for lists in (os.listdir(data_path)):
                            if lists.replace('json','jpg') in self.no_files:
                                # print("removing: ", lists.replace('json','jpg'))
                                continue
                            paths.append(os.path.join(data_path,lists.split('.json')[0]).replace('\\', '/'))
                elif 'ian' in dir:
                    continue
                else:
                    if os.path.exists(os.path.join(self.data_path,dir,'hdJoints')):
                        data_path = os.path.join(self.data_path,dir,'hdJoints')
                        for lists in (os.listdir(data_path)):
                            if lists.replace('json','jpg') in self.no_files:
                                # print("removing: ", lists.replace('json','jpg'))
                                continue
                            paths.append(os.path.join(data_path,lists.split('.json')[0]).replace('\\', '/'))

        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def get_second_view(self, image_path):
        """Randomly gets another camera view"""
        split = image_path.split(';')
        camera = split[1].split('_')
        view = int(camera[-1]) # id of the first view

        second_view = random.randint(0, view) if view > 15 else random.randint(view + 1, 31) # randomly get second view id that is smaller or bigger than the first one

        camera[2] = str(second_view) if second_view > 9 else '0' + str(second_view)
        camera = '_'.join(camera)
        split[1] = camera
        second_path = ';'.join(split)

        image_name = image_path.split('/')[-1]
        if image_name in self.no_files:
            return "Invalid Image"

        return second_path


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        path_split = self.data['paths'][idx].split('/hdJoints')
        image1_path = path_split[0] + '/hdImages' + path_split[-1] + '.jpg'
        image2_path = self.get_second_view(image1_path)

        for i in range(0, 20):
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

class CompleteContrastivePanopticDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", drop = 0.5):

        #open green_images.txt
        self.no_files = []
        with open(dataset_dir+"/green_images.txt") as f:
            for line in f:
                self.no_files.append(line.strip())

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/ProcessedPanopticDataset/"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['scripts','python','matlab','.git','glViewer.py','README.md','matlab',
                'README_kinoptic.md', '171204_pose3']
        
        print("Loading the sequences")
        for dir in tqdm(motion_seq):
            if dir not in no_dir:
                if 'haggling' in dir:
                    continue
                elif dir == '171204_pose2' or dir =='171204_pose5' or dir =='171026_cello3':
                    if os.path.exists(os.path.join(self.data_path,dir, 'hdJoints')):
                        data_path = os.path.join(self.data_path,dir, 'hdJoints')
                        for lists in (os.listdir(data_path)):
                            if lists.replace('json','jpg') in self.no_files:
                                # print("removing: ", lists.replace('json','jpg'))
                                continue
                            paths.append(os.path.join(data_path,lists.split('.json')[0]).replace('\\', '/'))
                elif 'ian' in dir:
                    continue
                else:
                    if os.path.exists(os.path.join(self.data_path,dir,'hdJoints')):
                        data_path = os.path.join(self.data_path,dir,'hdJoints')
                        for lists in (os.listdir(data_path)):
                            if lists.replace('json','jpg') in self.no_files:
                                # print("removing: ", lists.replace('json','jpg'))
                                continue
                            paths.append(os.path.join(data_path,lists.split('.json')[0]).replace('\\', '/'))
                            

        print("generating data pairs")
        # generate all possible pairs of images from different camera views if they belong to the same sequence
        self.pairs = {}
        for path in tqdm(paths):
            found_pair = False
            # try all possible views
            for i in range(0, 31):
                # get the second view
                split = path.split('/hdJoints')
                image1_path = split[0] + '/hdImages' + split[-1] + '.jpg'

                id = image1_path.split(';')
                id.pop(1)
                id = ';'.join(id)
                
                camera = image1_path.split(';')[1].split('_')
                camera[2] = str(i) if i > 9 else '0' + str(i)
                camera = '_'.join(camera)
                image2_path = image1_path.split(';')[0] + ';' + camera + ";" + ";".join(image1_path.split(';')[2:])

                if (os.path.isfile(image2_path) and image1_path != image2_path):
                    if id not in self.pairs:
                        self.pairs[id] = [(image1_path, image2_path)]
                    elif( (image1_path, image2_path) not in self.pairs[id] and (image2_path, image1_path) not in self.pairs[id]):
                        self.pairs[id].append((image1_path, image2_path))
                        
                    found_pair = True
            if not found_pair:
                # if no pair is found, apply random rotation on the first image
                self.pairs[id] = [(image1_path, image1_path)]
            # breakpoint()
        
        # drop randomly pairs from each id
        for id in self.pairs:
            self.pairs[id] = random.sample(self.pairs[id], math.ceil(len(self.pairs[id]) * (1 - drop)))

        self.pairs = [(image1, image2) for id in self.pairs for image1, image2 in self.pairs[id]]

        # breakpoint()

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image1_path, image2_path = self.pairs[idx]

        image1 = cv2.imread(image1_path)
        image1 =cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = self.transform(image1)

        image2 = cv2.imread(image2_path)
        image2 =cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = self.transform(image2)

        if image1_path == image2_path:
            # apply random rotation on the first image if the second view is the same as the first one
            image1 = T.RandomRotation(45)(image1)

        sample['image1'] = image1
        sample['image2'] = image2
        # breakpoint()

        return sample



def getContrastiveDatasetPanoptic(transform, dataset_dir="datasets", mode="complete", drop=0.5):
    """
    Returns training and validation datasets for clustering in the Panoptic dataset.

    Args:
        transform (callable): A function/transform that takes in an image and its annotations and returns a transformed version.
        dataset_dir (str, optional): The directory where the dataset is located. Defaults to "datasets".
        mode (str, optional): The mode of the dataset. Can be "complete" or "simple". Defaults to "complete".
        drop (float, optional): The percentage of pairs to drop. Defaults to 0.5.

    Returns:
        training_data (torch.utils.data.Dataset): The training dataset.
        val_data (torch.utils.data.Dataset): The validation dataset.
    """
    generator.manual_seed(0)
    if mode == "complete":
        dataset = CompleteContrastivePanopticDataset(transform, dataset_dir, drop)
    elif mode == "simple":
        dataset = ContrastivePanopticDataset(transform, dataset_dir)
    else:
        raise ValueError("Invalid mode. Choose between 'complete' and 'simple'.")

    num_samples = len(dataset)

    training_samples = int(num_samples * 0.6 + 1)
    val_samples = int(num_samples * 0.2 + 1)
    test_samples = num_samples - training_samples - val_samples

    training_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [training_samples, val_samples, test_samples], generator=generator
    )

    return training_data, val_data, test_data


class ClusterPanopticDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", set = ''):

        #open green_images.txt
        no_files = []
        with open(dataset_dir+"/green_images.txt") as f:
            for line in f:
                no_files.append(line.strip())

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/ProcessedPanopticDataset/171204_pose3/hdImages"

        images = [os.path.join(self.data_path, f).replace('\\','/') for f in os.listdir(self.data_path) 
                    if (os.path.isfile(os.path.join(self.data_path, f)) and f not in no_files)] [0:6000]
        print("Number of images: ", len(images))
        self.transform = transform

        self.data = {'paths': images}


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
        sample['path'] = image_path

        return sample

class PosePanopticDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", use_cluster="NONE"):

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/ProcessedPanopticDataset"
        self.training_dir = []

        self.transform = transform

        paths = []

        included_images = []

        if not use_cluster.startswith("RANDOM") and use_cluster != "NONE":
            with open(use_cluster, 'r') as f:
                for line in f:
                    included_images.append(line.strip())
                
        #open green_images.txt
        no_files = []
        with open(dataset_dir+"/green_images.txt") as f:
            for line in f:
                no_files.append(line.strip())

        motion_seq = os.listdir(self.data_path)
        no_dir = ['scripts','python','matlab','.git','glViewer.py','README.md','matlab',
                'README_kinoptic.md']

        for dir in motion_seq:
            if dir not in no_dir:
                if 'haggling' in dir or 'ian' in dir:
                    continue
                else:
                    joint_path = os.path.join(self.data_path,dir,'hdJoints').replace('\\', '/')
                    if os.path.exists(joint_path):
                        for lists in (os.listdir(joint_path)):
                            if not lists.replace('json','jpg') in no_files:
                                if len(included_images) == 0:
                                    paths.append(os.path.join(joint_path,lists.split('.json')[0]).replace('\\', '/'))
                                elif lists in included_images:
                                    for i in range(included_images.count(lists)):
                                        paths.append(os.path.join(joint_path,lists.split('.json')[0]).replace('\\', '/'))

        self.data = {'paths': paths}

        if use_cluster.startswith("RANDOM"):
            percent = int(use_cluster.split("_")[-1])
            self.data['paths'] = random.sample(self.data['paths'], math.ceil(len(self.data['paths']) * percent / 100))

    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        path_split = self.data['paths'][idx].split('/hdJoints')
        image_path = path_split[0] + '/hdImages' + path_split[-1] + '.jpg'

        image = cv2.imread(image_path)
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        sample['image'] = image

        joints_path = self.data['paths'][idx]+'.json'

        with open(joints_path) as dfile:
            bframe = json.load(dfile)

        poses_3d = torch.tensor(np.array(bframe['poses_3d']), dtype=torch.float32)
        #remove every 4th element of one dimensional array poses_3d
        poses_3d = poses_3d.reshape(-1, 4)[:, :3].reshape(-1)

        sample['poses_3d'] =  poses_3d

        cam = bframe['cam']
        # Replace 'array' with 'list' in the cam string
        cam = cam.replace('array', 'list')

        # Load camera parameters
        camera_params = eval(cam)

        # Camera intrinsic matrix
        K = camera_params['K']

        # Rotation matrix and translation vector
        R = camera_params['R']
        t = camera_params['t']

        sample['cam'] = {'K':K, 'R':R, 't':t}

        return sample
    
def getPoseDatasetPanoptic(transform, dataset_dir="datasets", use_cluster="NONE"):
    """
    Returns training and validation datasets for pose estimation in the Panoptic dataset.

    Args:
        transform (callable): A function/transform that takes in an image and its annotations and returns a transformed version.
        dataset_dir (str, optional): The directory where the dataset is located. Defaults to "datasets".
        use_cluster (str, optional): The file containing the list of images to use. Defaults to "NONE" (use all images). RANDOM_percent will use a random percent of the images.

    Returns:
        training_data (torch.utils.data.Dataset): The training dataset.
        val_data (torch.utils.data.Dataset): The validation dataset.
    """
    generator.manual_seed(0)
    dataset = PosePanopticDataset(transform, dataset_dir, use_cluster)

    num_samples = len(dataset)

    training_samples = int(num_samples * 0.7 + 1)
    val_samples = int(num_samples * 0.15 + 1)
    test_samples = num_samples - training_samples - val_samples

    training_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [training_samples, val_samples, test_samples], generator=generator
    )

    return training_data, val_data, test_data

class ContrastivePanopticDatasetMoco(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):
        self.no_files = []
        with open(dataset_dir+"/green_images.txt") as f:
            for line in f:
                self.no_files.append(line.strip())

        # change this to the path where the dataset is stored
        self.data_path = dataset_dir+"/ProcessedPanopticDataset/"
        self.training_dir = []

        self.transform = transform

        paths = []

        motion_seq = os.listdir(self.data_path)
        no_dir = ['scripts','python','matlab','.git','glViewer.py','README.md','matlab',
                'README_kinoptic.md', '171204_pose3']

        for dir in motion_seq:
            if dir not in no_dir:
                if 'haggling' in dir:
                    continue
                elif dir == '171204_pose2' or dir =='171204_pose5' or dir =='171026_cello3':
                    if os.path.exists(os.path.join(self.data_path,dir, 'hdJoints')):
                        data_path = os.path.join(self.data_path,dir, 'hdJoints')
                        for lists in (os.listdir(data_path)):
                            if lists.replace('json','jpg') in self.no_files:
                                print("removing: ", lists.replace('json','jpg'))
                                continue
                            paths.append(os.path.join(data_path,lists.split('.json')[0]).replace('\\', '/'))
                elif 'ian' in dir:
                    continue
                else:
                    if os.path.exists(os.path.join(self.data_path,dir,'hdJoints')):
                        data_path = os.path.join(self.data_path,dir,'hdJoints')
                        for lists in (os.listdir(data_path)):
                            if lists.replace('json','jpg') in self.no_files:
                                print("removing: ", lists.replace('json','jpg'))
                                continue
                            paths.append(os.path.join(data_path,lists.split('.json')[0]).replace('\\', '/'))

        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def get_all_view(self, image_path):
        """Gets every camera view of the image"""
        split = image_path.split(';')
        camera = split[1].split('_')
        view = int(camera[-1]) # id of the first view

        paths = []

        for i in range(0, 30):
          if view != i:
            camera[2] = str(i) if i > 9 else '0' + str(i)
            path = '_'.join(camera)
            split[1] = path
            new_path = ';'.join(split)
            image_name = image_path.split('/')[-1]
            if image_name not in self.no_files:
                paths.append(new_path)

        return paths


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()
        keys = []

        path_split = self.data['paths'][idx].split('/hdJoints')
        query_path = path_split[0] + '/hdImages' + path_split[-1] + '.jpg'
        keys_path = self.get_all_view(query_path)

        for i in range(0, len(keys_path)):
            if os.path.isfile(keys_path[i]):
                img = cv2.imread(keys_path[i])
                img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
                keys.append(img)


        query = cv2.imread(query_path)
        query =cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        query = self.transform(query)

        sample['query'] = query
        sample['keys'] = keys

        return sample

def getContrastiveDatasetPanopticMoco(transform, dataset_dir="datasets"):
    """
    Returns training and validation datasets for clustering in the Panoptic dataset.

    Args:
        transform (callable): A function/transform that takes in an image and its annotations and returns a transformed version.
        dataset_dir (str, optional): The directory where the dataset is located. Defaults to "datasets".

    Returns:
        training_data (torch.utils.data.Dataset): The training dataset.
        val_data (torch.utils.data.Dataset): The validation dataset.
    """
    
    dataset = ContrastivePanopticDatasetMoco(transform, dataset_dir)
    
    num_samples = len(dataset)

    training_samples = int(num_samples * 0.6 + 1)
    val_samples = int(num_samples * 0.2 + 1)
    test_samples = num_samples - training_samples - val_samples

    training_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [training_samples, val_samples, test_samples], generator=generator
    )

    return training_data, val_data, test_data
