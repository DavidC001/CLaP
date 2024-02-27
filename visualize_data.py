import os
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import cv2
import re
import random
import torchvision.transforms as transforms
from torchvision.io import read_image

import matplotlib.pyplot as plt
import torchvision.transforms as T

'''
Define the SimCLR base model with encoder and projection head
'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return x, self.layers(x)
    
from torchvision.models import resnet50, ResNet50_Weights


def get_simclr_net():
    """
    Returns the SimCLR network model.

    This function creates a SimCLR network model by using a pre-trained ResNet50 model
    as the backbone and replacing the fully connected layer with a custom MLP layer.

    Returns:
        model (nn.Module): SimCLR network model.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = MLP(2048, 2048, 128)

    return model

'''
Define the dataloader for the clustering task
'''
class ClusterDataset(Dataset):
    def __init__(self, transform, data_set='training'):

        # change this to the path where the dataset is stored
        self.data_path = "ProcessedPanopticDataset/171204_pose3/hdImages"

        images = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))][0:6000]
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

        return sample

def get_cluster_data(batch_size):
    """
    Get the cluster dataset and data loader.

    Args:
        batch_size (int): The batch size for the data loader.

    Returns:
        cluster_dataset (ClusterDataset): The cluster dataset.
        cluster_loader (DataLoader): The data loader for the cluster dataset.
    """
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(128, 128)),
        ]
    )

    cluster_dataset = ClusterDataset(transforms)
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size)

    return cluster_dataset, cluster_loader


def extract_representations(path, cluster_loader, load=True):
    """
    Extracts representations from images using a SimCLR network.

    Args:
        path (str): The path to the saved model checkpoint.
        cluster_loader (torch.utils.data.DataLoader): The data loader for the images.
        load (bool, optional): Whether to load the model weights from the checkpoint. 
            Defaults to True.

    Returns:
        numpy.ndarray: The concatenated base representations.
        numpy.ndarray: The concatenated projected representations.
    """
    net = get_simclr_net()

    if load:
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    net.to('cpu')
    net.eval()

    proj_repr = []
    base_repr = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(cluster_loader):
            images = inputs['image']
            images.to('cpu')
            base, proj = net(images)
            proj_repr.append(proj)
            base_repr.append(base)

    return torch.cat(base_repr).numpy(), torch.cat(proj_repr).numpy()


from sklearn.cluster import KMeans

def kmeans_algorithm(features, n_clusters=8):
    """
    Applies the K-means algorithm to the given features.

    Parameters:
        features (array-like): The input features to be clustered.
        n_clusters (int): The number of clusters to create (default is 8).

    Returns:
        array-like: The cluster labels assigned to each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=3)
    # 1 is good
    # 2 3 is less visually appealing, but more similar to one on slides
    # 5 is close to slides
    # 4 shows green cluster
    # 6 is the one used in slide to show green cluster
    kmeans.fit(features)

    return kmeans.labels_


#import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def reduce_dim(features, labels):
    """
    Reduce the dimensionality of the input features using Linear Discriminant Analysis (LDA).

    Parameters:
        features (array-like): The input features.
        labels (array-like): The corresponding labels for the features.

    Returns:
        array-like: The transformed features with reduced dimensionality.
    """
    lda = LDA(n_components=3)
    lda.fit(features, labels)

    #save transformation matrix to file for later use
    with open('lda_transform.npy', 'wb') as f:
        np.save(f, lda.scalings_)

    return lda.transform(features)

def plot_clusters(dataset, clusters, features, title):
    """
    Plots clusters in a 3D scatter plot and displays corresponding images on selection.

    Parameters:
    - dataset (Dataset): The dataset containing the image paths.
    - clusters (list): List of cluster labels for each data point.
    - features (ndarray): Array of features for each data point.
    - title (str): Title of the plot.

    Returns:
    None
    """
    colors = {
        0: '#F8512E', 1: '#F8F82E',
        2: '#40F82E', 3: '#2EC1F8',
        4: '#6B2EF8', 5: '#D92EF8',
        6: '#731642', 7: '#092040'
    }

    cluster_colors = [colors[c] for c in clusters]

    hover_images = [dataset.data['paths'][i] for i in range(len(dataset.data['paths']))]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121, projection='3d')  # Original 3D scatter plot
    img_ax = fig.add_subplot(122)  # New subplot for displaying the image

    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=cluster_colors, picker=True)

    def onpick(event):
        ind = event.ind[0]
        try:
            img = cv2.imread(hover_images[ind])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_ax.imshow(img)
            plt.show()
        except Exception as e:
            print(f"Error loading image: {e}")

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.title(title)
    plt.show()

from sklearn.metrics import silhouette_score

#save pointcloud as mesh with spheres for each point with color corresponding to cluster
def save_pointcloud(features, clusters, path):
    """
    Save a point cloud as a mesh with spheres for each point and color corresponding to the cluster.

    Parameters:
    - features (ndarray): The features for each data point.
    - clusters (list): The cluster labels for each data point.
    - path (str): The path to save the mesh.

    Returns:
    None
    """
    from pyntcloud import PyntCloud
    import pandas as pd

    colors = {
        0: [248, 82, 46], 1: [248, 248, 46],
        2: [64, 248, 46], 3: [46, 193, 248],
        4: [107, 46, 248], 5: [217, 46, 248],
        6: [115, 22, 66], 7: [9, 32, 64]
    }

    color_list = [colors[c] for c in clusters]
    color_list = np.array(color_list) / 255.0

    df = pd.DataFrame(features, columns=['x', 'y', 'z'])
    df['red'] = color_list[:, 0]
    df['green'] = color_list[:, 1]
    df['blue'] = color_list[:, 2]

    cloud = PyntCloud(df)

    cloud.to_file(path)

def cluster(model_path, load=True):
    """
    Clusters the data using the given model and returns the cluster set, base features, projected features,
    labels for base features, labels for projected features, reduced base features, and reduced projected features.

    Parameters:
    - model_path (str): The path to the model file.
    - load (bool): Whether to load the model or not. Defaults to True.

    Returns:
    - cluster_set: The cluster set.
    - base_features: The base features.
    - proj_features: The projected features.
    - labels_base: The labels for base features.
    - labels_proj: The labels for projected features.
    - lda_base: The reduced base features.
    - lda_proj: The reduced projected features.
    """
    cluster_set, cluster_loader = get_cluster_data(30)

    base_features, proj_features = extract_representations(model_path, cluster_loader, load)

    labels_base = kmeans_algorithm(base_features)
    lda_base = reduce_dim(base_features, labels_base)

    labels_proj = kmeans_algorithm(proj_features)
    lda_proj = reduce_dim(proj_features, labels_proj)

    silhouette_base = silhouette_score(base_features, labels_base)
    silhouette_proj = silhouette_score(proj_features, labels_proj)

    print("Silhouette score for the encoder features: {}".format(silhouette_base))
    print("Silhouette score for the projection head features: {}".format(silhouette_proj))

    plot_clusters(cluster_set, labels_base, lda_base, "Encoder features")
    plot_clusters(cluster_set, labels_proj, lda_proj, "Projection head features")

    save_pointcloud(lda_base, labels_base, "base_features.ply")
    save_pointcloud(lda_proj, labels_proj, "proj_features.ply")

    return cluster_set, base_features, proj_features, labels_base, labels_proj, lda_base, lda_proj

#get the latest model
path = 'trained_models/simclr/'
epoch = 0
for file in os.listdir(path):
    if 'simclr_epoch' in file:
        e = int(re.findall(r'\d+', file)[0])
        if e > epoch:
            epoch = e

epoch = 22
path = path + 'simclr_epoch_{:d}.pth'.format(epoch)

#path = 'trained_models/ver1.pt'

#cluster the data
cluster_set, base_features, proj_features, labels_base, labels_proj, pca_base, pca_proj = cluster(path)