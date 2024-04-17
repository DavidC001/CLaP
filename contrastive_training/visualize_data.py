import os
import torch
import torch.nn as nn
import cv2
import re

import matplotlib.pyplot as plt
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "panoptic"
path = 'trained_models/simclr_both_biggerBatch/'
#both_bigger batch huge bias on skin color!


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
    model = nn.DataParallel(model)

    return model

'''
Define the dataloader for the clustering task
'''
import sys
sys.path.append('.')
from dataloaders.datasets import cluster_datasets

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

    cluster_dataset = cluster_datasets[dataset](transform=transforms)
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
        net.load_state_dict(torch.load(path, map_location=torch.device(device)))

    net.to(device)
    net.eval()

    proj_repr = []
    base_repr = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(cluster_loader):
            images = inputs['image']
            images.to(device)
            base, proj = net(images)
            proj_repr.append(proj)
            base_repr.append(base)

    return torch.cat(base_repr).cpu().numpy(), torch.cat(proj_repr).cpu().numpy()


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
#import PCA
from sklearn.decomposition import PCA

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

    #pca = PCA(n_components=3)
    #pca.fit(features)

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

    return cluster_set, base_features, proj_features, labels_base, labels_proj, lda_base, lda_proj

#get the latest model
epoch = 0
for file in os.listdir(path):
    if 'epoch' in file:
        e = int(re.findall(r'\d+', file)[0])
        if e > epoch:
            epoch = e


path = path + 'epoch_{:d}.pth'.format(epoch)

#path = 'trained_models/ver1.pt'

#cluster the data
cluster_set, base_features, proj_features, labels_base, labels_proj, pca_base, pca_proj = cluster(path)