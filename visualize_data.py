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
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = MLP(2048, 2048, 128)

    return model

from flash.core.optimizers import LARS


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
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)

    return kmeans.labels_


#import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def reduce_dim(features, labels):
    lda = LDA(n_components=3)
    lda.fit(features, labels)

    return lda.transform(features)

def plot_clusters(dataset, clusters, features, title):
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


def cluster(model_path, load=True):
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

    return cluster_set, base_features, proj_features, labels_base, labels_proj, lda_base, lda_proj

#get the latest model
path = 'trained_models/simclr/'
epoch = 0
for file in os.listdir(path):
    if 'simclr_epoch' in file:
        e = int(re.findall(r'\d+', file)[0])
        if e > epoch:
            epoch = e

path = path + 'simclr_epoch_{:d}.pth'.format(epoch)

cluster_set, base_features, proj_features, labels_base, labels_proj, pca_base, pca_proj = cluster(path)