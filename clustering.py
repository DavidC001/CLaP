import torch
from torchvision import transforms as T
import numpy as np
from sklearn.cluster import KMeans
from dataloaders.datasets import cluster_datasets
from torchvision.models.resnet import  ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm
from torchvision.models.resnet import resnet50

#wrap function to get selected images from clusters
def get_selected_images(model, base_model, dataset, dataset_dir, name_file, device = 'cuda'):
    cluster_data = get_dataSet(dataset, dataset_dir, base_model)
    representations = extract_representations(model, cluster_data)
    kmeans = kmeans_clustering(representations, 10)
    selected_images = select_images_from_clusters(kmeans, cluster_data, 10)
    write_to_file(selected_images, name_file)
    return selected_images

def extract_representations(model, loader):
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

def select_images_from_clusters(kmeans, dataset, n_clusters):
    labels = kmeans.labels_
    selected_images = []

    n = int(len(dataset) * 0.2 / n_clusters)

    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        selected_indices = np.random.choice(indices, size=n, replace=False)
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
    model.device = device
    base_model = 'resnet50'
    dataset = 'skiPose'
    dataset_dir = 'datasets'
    name_file = 'selected_images.txt'
    get_selected_images(model, base_model, dataset, dataset_dir, name_file)