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

generator = torch.Generator()

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

class MultiViewSkiDataset(Dataset):
    def __init__(self, transform, dataset_dir="datasets", mode="train", max_views=6, num_cameras=6, augmentation_degree=45):
        """
        Custom Dataset to retrieve multiple camera views for each frame, pad to `max_views` with rotation augmentation if needed,
        and shuffle the order of images.

        Args:
            transform (torchvision.transforms.Transform): Transformations to apply to the images.
            dataset_dir (str, optional): Directory where the dataset is stored. Defaults to "datasets".
            mode (str, optional): Mode of the dataset, either "train" or "test". Defaults to "train".
            max_views (int, optional): Maximum number of views per data point. Defaults to 5.
            num_cameras (int, optional): Total number of camera views available. Defaults to 6.
            augmentation_degree (int, optional): Degree for random rotation augmentation. Defaults to 45.
        """
        self.transform = transform
        self.max_views = max_views
        self.num_cameras = num_cameras
        self.augmentation_degree = augmentation_degree

        self.data_path = os.path.join(dataset_dir, "Ski-PosePTZ-CameraDataset-png")
        self.mode = mode

        self.frames = []  # List of lists, where each sublist contains image paths from different cameras for a frame

        # Define camera names based on the number of cameras
        self.cameras = [f"cam_{i:02d}" for i in range(self.num_cameras)]

        # Populate self.frames
        self._prepare_frames()

    def _prepare_frames(self):
        """
        Prepare the list of frames, each containing paths to images from different cameras.
        """
        if self.mode not in ['train', 'test']:
            raise ValueError("mode should be 'train' or 'test'")

        dir_mode = 'train' if self.mode == 'train' else 'test'
        mode_path = os.path.join(self.data_path, dir_mode)

        # Iterate through each sequence in the mode directory
        for seq in os.listdir(mode_path):
            seq_path = os.path.join(mode_path, seq)
            if not os.path.isdir(seq_path):
                continue  # Skip if not a directory

            # Iterate through each camera in the sequence
            frame_dict = {}  # key: frame identifier, value: list of image paths from different cameras
            for cam in self.cameras:
                cam_path = os.path.join(seq_path, cam)
                if not os.path.isdir(cam_path):
                    continue  # Skip if camera directory does not exist

                for img_file in os.listdir(cam_path):
                    frame_id = img_file  # Assuming the frame identifier is the image filename
                    img_path = os.path.join(cam_path, img_file).replace('\\', '/')
                    if frame_id not in frame_dict:
                        frame_dict[frame_id] = []
                    frame_dict[frame_id].append(img_path)

            # Add all frames from this sequence to the dataset
            for frame_images in frame_dict.values():
                self.frames.append(frame_images)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Retrieves a data point consisting of a list of images.

        Args:
            idx (int): Index of the data point.

        Returns:
            dict: Contains 'images' key with a list of transformed images.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_images = self.frames[idx].copy()  # List of image paths for this frame

        # If number of images > max_views, randomly select max_views images
        if len(frame_images) > self.max_views:
            frame_images = random.sample(frame_images, self.max_views)


        # Shuffle the order of images
        random.shuffle(frame_images)

        transformed_images = []
        for img_path in frame_images:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            transformed_images.append(image)

        # If number of images < max_views, pad with augmented images
        if len(transformed_images) < self.max_views:
            num_augmented = self.max_views - len(transformed_images)
            augmented_images = self._augment_images(frame_images, num_augmented)
            transformed_images.extend(augmented_images)
        
        return {'images': transformed_images}

    def _augment_images(self, existing_image_paths, num_augmented):
        """
        Generate augmented images by applying random rotations to existing images.

        Args:
            existing_image_paths (list): List of existing image paths.
            num_augmented (int): Number of augmented images to generate.

        Returns:
            list: List of augmented image tensors.
        """
        augmented = []
        for _ in range(num_augmented):
            # Select a random image to augment
            img_path = random.choice(existing_image_paths)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found for augmentation: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Apply random rotation
            rotated_image = self.transform(image)
            rotated_image = T.RandomRotation(self.augmentation_degree)(rotated_image)
            augmented.append(rotated_image)
        return augmented

def getContrastiveDatasetSki(transform, dataset_dir="datasets", mode="complete", drop=0.5):
    """
    Returns a tuple of train and test datasets for contrastive learning using Ski data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".
        mode (str, optional): The mode of the dataset. "simple" for simple contrastive learning, "complete" for all pairs "multi" for multiple views. Defaults to "complete".
        drop (float, optional): The percentage of pairs to drop. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    generator.manual_seed(42)
    if mode == "complete":
        dataset = CompleteContrastiveSkiDataset(transform, dataset_dir, mode="train", drop=drop)
        test = CompleteContrastiveSkiDataset(transform, dataset_dir, mode="test", drop=drop)
    elif mode == "simple":
        dataset = ContrastiveSkiDataset(transform, dataset_dir, mode="train")
        test = ContrastiveSkiDataset(transform, dataset_dir, mode="test")
    elif mode == "multi":
        dataset = MultiViewSkiDataset(transform, dataset_dir, mode="train")
        test = MultiViewSkiDataset(transform, dataset_dir, mode="test")
    else:
        raise ValueError("Invalid mode. Choose 'simple', 'complete', or 'multi'.")

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
        self.idxs = []

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
                for line in f:
                    included_images.append(line.strip())
        
        #load image's path in order
        for index in range(0,len(h5_label_file['cam'])):
            seq   = int(h5_label_file['seq'][index])
            cam   = int(h5_label_file['cam'][index])
            frame = int(h5_label_file['frame'][index])

            image_path = data_path+dir+'/seq_{:03d}/cam_{:02d}/image_{:06d}.png'.format(seq,cam,frame)
            # breakpoint()
            if len(included_images) == 0:
                self.idxs.append(index)
                paths.append(image_path.replace('\\','/'))
            elif image_path in included_images:
                # repeat for the number of times the image is repeated in included_images
                for i in range(included_images.count(image_path)):
                    paths.append(image_path.replace('\\','/'))
                    self.idxs.append(index)

        if use_cluster.startswith("RANDOM"):
            percent = int(use_cluster.split("_")[-1])
            num_samples = len(paths)
            paths = random.sample(paths, math.ceil(num_samples * percent / 100))

        self.data = {'paths': paths}

    def __len__(self):
        return len(self.data['paths'])

    def __getitem__(self, idx):


        sample = dict()

        #read the image
        image = cv2.imread(self.data['paths'][idx])
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        sample['image'] = image
        
        #load the joints position
        path_file = self.data['paths'][idx].split('/seq')[0]+'/labels.h5'
        h5_label_file = h5py.File(path_file, 'r')
        poses_3d = (h5_label_file['3D'][self.idxs[idx]]).reshape([-1,3])

        sample['poses_3d'] =  poses_3d

        #camera param
        intrinsic = h5_label_file['cam_intrinsic'][self.idxs[idx]].reshape([-1,3])
        traslation = h5_label_file['cam_position'][self.idxs[idx]]
        rotation = h5_label_file ['R_cam_2_world'][self.idxs[idx]].reshape([3,3])
        cam = {'K':intrinsic, 'R':rotation, 't':traslation}

        sample['cam'] = cam

        return sample

def getPoseDatasetSki(transform, dataset_dir="datasets", use_cluster="NONE"):
    """
    Returns a tuple of train and test datasets for pose estimation using Ski data.

    Args:
        transform (torchvision.transforms.Transform): The data transformation to be applied to the dataset.
        dataset_dir (str, optional): The directory where the datasets are stored. Defaults to "datasets".
        use_cluster (str, optional): The file containing the list of images to use. Defaults to "NONE" (use all images). RANDOM_percent will use a random percent of the images

    Returns:
        tuple: A tuple containing the train and test datasets.
    """
    generator.manual_seed(42)
    train = PoseSkiDataset(transform, dataset_dir, mode="train", use_cluster=use_cluster)
    
    num_samples = len(train)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    train, val = torch.utils.data.random_split(
        train, [training_samples, val_samples], generator=generator
    )

    test = PoseSkiDataset(transform, dataset_dir, mode="test")
    
    return train, val, test

if __name__ == '__main__':

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor()
    ])
    train = PoseSkiDataset(transform, dataset_dir="datasets", mode="train", use_cluster = "selected_images.txt")
    print(len(train))
    print(train[0])

    with open("selected_images.txt", 'r') as f:
        selected_images = f.readlines()
    print(len(selected_images))
    print(selected_images[0])
