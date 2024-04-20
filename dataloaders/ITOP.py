import os
import torch
from torch.utils.data import Dataset
import cv2 as cv 
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
import h5py 


class ContrastiveKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):
        print("KinectDataset")
        
        data_path = dataset_dir+"/ITOP/side_" + mode + "_qqimages"
        self.training_dir = []
        self.transform = transform

        paths = []

        if mode == 'train':
            dir = '/train'
        else:
            dir = '/test'

        for img in (os.listdir(data_path)):
            if img.endswith('.jpg'):
                paths.append(os.path.join(data_path, img).replace('\\', '/'))
        
        self.data = {'paths': paths}

        
    def __len__(self):
        return len(self.data['paths'])
    

    def __getitem__(self, idx):
        image_path = self.data['paths'][idx]
        top_view_path = image_path.replace("side", "top")

        image_side = cv.imread(image_path)
        image_top = cv.imread(top_view_path)

        image_side = cv.cvtColor(image_side, cv.COLOR_BGR2RGB)
        image_top = cv.cvtColor(image_top, cv.COLOR_BGR2RGB)

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
    num_samples = len(train)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    train, val, = torch.utils.data.random_split(
        train, [training_samples, val_samples], generator=generator
    )

    test = ContrastiveKinectDataset(transform, dataset_dir, mode="test")
    return train, val, test



class ClusterKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets"):
        #print("ClusterKinectDataset")

        views = ['side', 'top']

        for view in views:
            self.data_path = dataset_dir+"/ITOP/"+view+"_test_images"
            self.transform = transform

            paths = []

            motion_seq = os.listdir(self.data_path)
                    

            for img in motion_seq:
                if img.endswith('.jpg'):
                    paths.append(os.path.join(self.data_path, img).replace('\\', '/'))
        
        self.data = {'paths': paths}

        
    def __len__(self):
        return len(self.data['paths'])
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()

        image_path = self.data['paths'][idx]

        image = cv.imread(image_path)
        image =cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = self.transform(image)

        sample['image'] = image

        return sample


class PoseKinectDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train"):

        self.transform = transform

        paths = {}

        if mode == 'train':
            dir = '/train'
        else:
            dir = '/test'
        
        views = ['side', 'top']

        for view in views:
            paths[view] = []
            data_path = dataset_dir+"/ITOP/ITOP_"+view+"_"+mode+"_labels.h5"
            h5_label_file = h5py.File(data_path, 'r')

            for id in h5_label_file['id']:
                # print(id)
                id = id.decode('utf-8')
                image_path = dataset_dir+"/ITOP/"+view+"_"+mode+"_images/"+str(id)+".jpg"
                if os.path.exists(image_path):
                    paths[view].append(image_path)
                else:
                    print("Image not found: ", image_path)

        self.data = {'paths': paths}
        self.mode = mode

    def __len__(self):
        return len(self.data['paths']['side']+self.data['paths']['top'])
    
    def __getitem__(self, idx):
        
        view = 'side'
        if idx >= len(self.data['paths']['side']):
            view = 'top'
            idx -= len(self.data['paths']['side'])

        image_path = self.data['paths'][view][idx]

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        root = image_path.split("/ITOP/")[0]
        path_file = root+"/ITOP/ITOP_"+view+"_"+self.mode+"_labels.h5"

        h5_label_file = h5py.File(path_file, 'r')
        joint_3d = h5_label_file["real_world_coordinates"][idx]

        sample = dict()
        sample['image'] = self.transform(image)
        sample['poses_3d'] = joint_3d
        sample['cam'] = None

        return sample
    

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
    num_samples = len(train)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    train, val, = torch.utils.data.random_split(
        train, [training_samples, val_samples], generator=generator
    )
    test = PoseKinectDataset(transform, dataset_dir, mode="test")
    return train, val, test


if __name__ == "__main__":
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # dataset = ContrastiveKinectDataset(transform)

    # for i in range(10):
    #     sample = dataset[i]
    #     image1 = sample['image1']
    #     image2 = sample['image2']
    #     plt.imshow(image1.permute(1, 2, 0))
    #     plt.show()
    #     plt.imshow(image2.permute(1, 2, 0))
    #     plt.show()
    
    # dataset = ClusterKinectDataset(transform)
    
    # for i in range(10):
    #     sample = dataset[i]
    #     image = sample['image']
    #     plt.imshow(image.permute(1, 2, 0))
    #     plt.show()

    dataset = PoseKinectDataset(transform)

    print(len(dataset))

    for i in range(3):
        sample = dataset[i]
        image = sample['image']
        joint_3d = sample['poses_3d']
        print(joint_3d)
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
    
    print("Done")