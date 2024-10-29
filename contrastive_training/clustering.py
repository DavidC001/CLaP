import torch
from torchvision import transforms as T
import numpy as np
from sklearn.cluster import KMeans
from dataloaders.datasets import cluster_datasets
from torchvision.models.resnet import  ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm
from torchvision.models.resnet import resnet50
import math
from copy import deepcopy

#wrap function to get selected images from clusters
def get_selected_images(trained_model, base_model, dataset, dataset_dir, name_file, n_clusters, percentage, device = 'cuda'):
    """
    Get selected images from clusters

    Parameters:
        trained_model (torch.nn.Module): Trained model
        base_model (str): Base model used to train the model
        dataset (str): Name of the dataset
        dataset_dir (str): Directory where the dataset is stored
        name_file (str): Name of the file to save the selected images
        n_clusters (int): Number of clusters
        percentage (float): Percentage of the dataset to select
        device (str): Device to run the model on
    """
    model = deepcopy(trained_model)
    model.module.fc = torch.nn.Identity()

    cluster_data = get_dataSet(dataset, dataset_dir, base_model)
    representations = extract_representations(model, cluster_data, device)
    kmeans = kmeans_clustering(representations, n_clusters)
    selected_images = select_images_from_clusters(kmeans, cluster_data, percentage)
    write_to_file(selected_images, name_file)
    return selected_images

def extract_representations(model, loader, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for data in tqdm(loader):
            images = data['image']
            images = images.to(device).unsqueeze(0)
            outputs = model(images).squeeze()
            representations.append(outputs.detach().cpu())
    return representations

def kmeans_clustering(representations, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(representations)
    return kmeans

def select_images_from_clusters(kmeans, dataset, percentage):
    labels = kmeans.labels_
    selected_images = []

    n = int(math.ceil(len(dataset) * (percentage/100) / len(np.unique(labels))))

    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        selected_indices = np.random.choice(indices, size=n, replace=True)
        selected_images.extend([dataset[i]['path'] for i in selected_indices])
    return selected_images

def write_to_file(selected_images, name_file):
    with open(name_file, 'w') as f:
        for image in selected_images:
            f.write(image + '\n')

    return selected_images

def get_dataSet(dataset, dataset_dir, base_model):

    if base_model == "resnet50":
        transforms = ResNet50_Weights.DEFAULT.transforms()
    elif base_model == "resnet18":
        transforms = ResNet18_Weights.DEFAULT.transforms()
    else:
        raise ValueError("Invalid base model")
    transforms = T.Compose([
        T.ToPILImage(), 
        transforms
    ])

    train = cluster_datasets[dataset](transforms, dataset_dir=dataset_dir, set = 'train')

    return train

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet50(pretrained=True).to(device)
    model = torch.nn.DataParallel(model)
    base_model = 'resnet50'
    dataset = 'skiPose'
    dataset_dir = 'datasets'
    name_file = 'selected_images.txt'
    get_selected_images(model, "simclr",base_model, dataset, dataset_dir, name_file, n_clusters=10, percentage=0.1, device=device)