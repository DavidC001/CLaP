"""
Complete Visualization Script for Clustering with SimCLR Features and GIF Creation
(Modified to show a bigger 3D plot on the left and a 2x2 grid of images on the right,
with thicker colored borders, updating clusters faster.)
"""

import os
import re
import random
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec
from collections import defaultdict

import torchvision.transforms as T
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --------------------
# Configuration
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_name = "skiPose"
backbone = "resnet18"  # Options: "resnet18", "resnet50"
model_path = 'trained_models/simclr_18_ski/'

# --------------------
# Define the SimCLR Model
# --------------------
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

def get_simclr_net(backbone_choice):
    """
    Returns the SimCLR network model based on the chosen backbone.

    Args:
        backbone_choice (str): "resnet18" or "resnet50"

    Returns:
        model (nn.Module): SimCLR network model.
    """
    if backbone_choice == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif backbone_choice == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        raise ValueError("Unsupported backbone. Choose 'resnet18' or 'resnet50'.")

    emb_dim = model.fc.in_features
    model.fc = MLP(emb_dim, emb_dim, 128)
    model = nn.DataParallel(model)

    return model

# --------------------
# Define the Dataset and DataLoader
# --------------------
import sys
sys.path.append('.')
from dataloaders.datasets import cluster_datasets  # Ensure this module is available

def get_cluster_data(dataset_name, batch_size):
    """
    Get the cluster dataset and data loader.

    Args:
        dataset_name (str): Name of the dataset.
        batch_size (int): The batch size for the data loader.

    Returns:
        cluster_dataset (Dataset): The cluster dataset.
        cluster_loader (DataLoader): The data loader for the cluster dataset.
    """
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(128, 128)),
        ]
    )

    cluster_dataset = cluster_datasets[dataset_name](transform=transforms)
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=batch_size)

    return cluster_dataset, cluster_loader

# --------------------
# Feature Extraction
# --------------------
def extract_representations(path, cluster_loader, device, load=True):
    """
    Extracts representations from images using a SimCLR network.

    Args:
        path (str): The path to the saved model checkpoint.
        cluster_loader (torch.utils.data.DataLoader): The data loader for the images.
        device (torch.device): The device to run the model on.
        load (bool, optional): Whether to load the model weights from the checkpoint. 
            Defaults to True.

    Returns:
        numpy.ndarray: The concatenated base representations.
        numpy.ndarray: The concatenated projected representations.
    """
    net = get_simclr_net(backbone)
    if load:
        net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    net.to(device)
    net.eval()

    proj_repr = []
    base_repr = []

    with torch.no_grad():
        for batch_idx, inputs in enumerate(cluster_loader):
            images = inputs['image'].to(device)
            base, proj = net(images)
            proj_repr.append(proj)
            base_repr.append(base)

    return torch.cat(base_repr).cpu().numpy(), torch.cat(proj_repr).cpu().numpy()

# --------------------
# Clustering Algorithms
# --------------------
def kmeans_algorithm(features, n_clusters=8):
    """
    Applies the K-means algorithm to the given features.

    Parameters:
        features (array-like): The input features to be clustered.
        n_clusters (int): The number of clusters to create (default is 8).

    Returns:
        array-like: The cluster labels assigned to each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_

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
    return lda.transform(features)

# --------------------
# Clustering Pipeline
# --------------------
def cluster_data(model_path, dataset_name, backbone, device, batch_size=30):
    """
    Clusters the data using the given model and returns the cluster set, base features, projected features,
    labels for base features, labels for projected features, reduced base features, and reduced projected features.

    Parameters:
        model_path (str): The path to the model file.
        dataset_name (str): Name of the dataset.
        backbone (str): Backbone model type ("resnet18" or "resnet50").
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for data loading.

    Returns:
        tuple: Contains cluster_set, base_features, proj_features, labels_base, labels_proj, lda_base, lda_proj
    """
    cluster_set, cluster_loader = get_cluster_data(dataset_name, batch_size)
    base_features, proj_features = extract_representations(model_path, cluster_loader, device, load=True)

    labels_base = kmeans_algorithm(base_features)
    lda_base = reduce_dim(base_features, labels_base)

    labels_proj = kmeans_algorithm(proj_features)
    lda_proj = reduce_dim(proj_features, labels_proj)

    silhouette_base = silhouette_score(base_features, labels_base)
    silhouette_proj = silhouette_score(proj_features, labels_proj)

    print("Silhouette score for the encoder features: {:.4f}".format(silhouette_base))
    print("Silhouette score for the projection head features: {:.4f}".format(silhouette_proj))

    return cluster_set, base_features, proj_features, labels_base, labels_proj, lda_base, lda_proj

# --------------------
# Updated Visualization (2x2 grid, bigger plot, faster)
# --------------------
def create_cluster_gif(
    cluster_set,
    features_3d,
    labels,
    cluster_colors,
    gif_path='clusters_rotation.gif',
    fps=20,
    seconds_per_cluster=3  # Reduced from 5s to 3s for faster updates
):
    """
    Creates and saves an animated GIF showing a rotating 3D scatter plot of clusters on the left
    (bigger) and a 2x2 grid of images on the right, each with a thick color border matching the cluster.

    Args:
        cluster_set (Dataset): Your dataset with .data['paths'] for image paths.
        features_3d (ndarray): 3D coordinates of the features.
        labels (ndarray): Cluster labels for each feature.
        cluster_colors (dict): Mapping from cluster index to color hex code.
        gif_path (str): Path to save the generated GIF.
        fps (int): Frames per second for the GIF.
        seconds_per_cluster (int): Duration each cluster's images are displayed.
    """
    from mpl_toolkits.mplot3d import Axes3D  # just to ensure 3D projection is recognized

    # 1) Group image paths by cluster and pick 4 examples per cluster
    cluster_to_paths = defaultdict(list)
    all_paths = cluster_set.data['paths']  # Ensure your dataset has .data['paths']

    for i, label in enumerate(labels):
        cluster_to_paths[label].append(all_paths[i])

    # For each cluster, pick exactly 4 images
    cluster_examples = {}
    for cluster_idx, paths in cluster_to_paths.items():
        if len(paths) >= 4:
            sample_paths = random.sample(paths, 4)
        else:
            # If fewer than 4 in a cluster, repeat images as needed
            sample_paths = paths * (4 // len(paths) + 1)
            sample_paths = sample_paths[:4]
        cluster_examples[cluster_idx] = sample_paths

    # Pre-load the images as RGB arrays
    cluster_images = {}
    for cidx, path_list in cluster_examples.items():
        imgs = []
        for p in path_list:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Image not found at path: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        cluster_images[cidx] = imgs

    # 2) Create figure with GridSpec
    # Make the left column wider for a bigger 3D plot
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1.3, 1, 1],  # left column is bigger
        wspace=0.15,
        hspace=0.15
    )

    # The 3D plot spans both rows in the first column
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')

    # 2x2 grid for images in the remaining columns
    ax_img1 = fig.add_subplot(gs[0, 1])
    ax_img2 = fig.add_subplot(gs[0, 2])
    ax_img3 = fig.add_subplot(gs[1, 1])
    ax_img4 = fig.add_subplot(gs[1, 2])
    image_axes = [ax_img1, ax_img2, ax_img3, ax_img4]

    # 3) Scatter the 3D points
    point_colors = [cluster_colors[l] for l in labels]

    scatter = ax3d.scatter(
        features_3d[:, 0],
        features_3d[:, 1],
        features_3d[:, 2],
        c=point_colors,
        s=30
    )

    ax3d.set_title("cluster visualization", fontsize=12)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # 4) Animation logic
    unique_clusters = sorted(cluster_images.keys())
    num_clusters = len(unique_clusters)
    frames_per_cluster = fps * seconds_per_cluster
    total_frames = frames_per_cluster * num_clusters

    def init():
        """Initialization function for FuncAnimation."""
        # Display the first cluster's images so we have something at the start
        for ax in image_axes:
            ax.clear()
            ax.axis('off')

        first_cluster = unique_clusters[0]
        color_border = cluster_colors[first_cluster]
        example_imgs = cluster_images[first_cluster]
        for ax_img, img_array in zip(image_axes, example_imgs):
            ax_img.imshow(img_array)
            ax_img.axis('off')
            # Thicker colored borders
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color_border)
                spine.set_linewidth(4)

        return [scatter] + image_axes

    def update(frame):
        """Update function for FuncAnimation, called for each frame."""
        # 1) Rotate the 3D plot
        rotation_angle = 360.0 * (frame / total_frames)
        ax3d.view_init(elev=30, azim=rotation_angle)

        # 2) Determine which cluster is active in this frame
        cluster_index = (frame // frames_per_cluster) % num_clusters
        active_cluster = unique_clusters[cluster_index]
        color_border = cluster_colors[active_cluster]

        # 3) Update the 4 images with the active cluster
        example_imgs = cluster_images[active_cluster]
        for ax_img, img_array in zip(image_axes, example_imgs):
            ax_img.clear()
            ax_img.imshow(img_array)
            ax_img.axis('off')
            # Thicker border to match the cluster color
            for spine in ax_img.spines.values():
                spine.set_edgecolor(color_border)
                spine.set_linewidth(4)

        return [scatter] + image_axes

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=False,       
        interval=1000/fps  # ms per frame
    )

    # Save to GIF
    ani.save(gif_path, writer='pillow', fps=fps)
    print(f"GIF saved to {gif_path}")

    plt.close(fig)

# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    # 1) Find the latest model checkpoint
    latest_epoch = -1
    for file in os.listdir(model_path):
        if 'epoch' in file:
            epochs_in_file = re.findall(r'\d+', file)
            if epochs_in_file:
                e = int(epochs_in_file[0])
                if e > latest_epoch:
                    latest_epoch = e

    if latest_epoch == -1:
        raise FileNotFoundError(f"No epoch checkpoint found in {model_path}")

    checkpoint_path = os.path.join(model_path, f'epoch_{latest_epoch}.pt')
    print(f"Using model checkpoint: {checkpoint_path}")

    # 2) Perform clustering
    cluster_set, base_features, proj_features, labels_base, labels_proj, lda_base, lda_proj = cluster_data(
        checkpoint_path,
        dataset_name,
        backbone,
        device,
        batch_size=30
    )

    # 3) Define cluster colors (adjust or extend if you have more than 8 clusters)
    cluster_colors = {
        0: '#F8512E', 1: '#F8F82E',
        2: '#40F82E', 3: '#2EC1F8',
        4: '#6B2EF8', 5: '#D92EF8',
        6: '#731642', 7: '#092040'
    }

    # If the data yields more clusters than we have colors for, add more randomly
    num_clusters_found = len(set(labels_base))
    if num_clusters_found > len(cluster_colors):
        import matplotlib.colors as mcolors
        additional_colors = list(mcolors.CSS4_COLORS.values())
        random.shuffle(additional_colors)
        idx_for_new = 0
        for new_cluster_idx in range(len(cluster_colors), num_clusters_found):
            cluster_colors[new_cluster_idx] = additional_colors[idx_for_new]
            idx_for_new += 1

    # 4) Create and save the GIF using LDA-reduced features, at a faster cluster switch
    create_cluster_gif(
        cluster_set=cluster_set,
        features_3d=lda_base,
        labels=labels_base,
        cluster_colors=cluster_colors,
        gif_path='clusters_rotation.gif',
        fps=20,               # frames per second
        seconds_per_cluster=3 # each cluster shown for 3s (faster than 5s)
    )

    print("Visualization complete.")
