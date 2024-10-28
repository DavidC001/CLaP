import torch
import torch.nn as nn
import torchvision.transforms as T
from dataloaders.datasets import contrastive_datasets
from dataloaders.datasets import combineDataSets
from dataloaders.datasets import moco_datasets
import os
import math

#import RN50 transforms
from torchvision.models.resnet import  ResNet18_Weights, ResNet50_Weights

train_data, val_data, test_data = None, None, None

def load_datasets(datasets, use_complete, drop,dataset_dir="datasets", base_model="resnet18"):
    """
    Prepares the datasets for contrastive training.

    Parameters:
        datasets: list, datasets to load
        use_complete: bool, use complete pairs
        drop: float, drop pairs
        dataset_dir: str, directory to save the datasets, default is 'datasets'
        base_model: str, base model, default is 'resnet18'
    """
    global train_data, val_data, test_data

    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms = T.Compose([T.ToPILImage(),
    #                         T.Resize(256),
    #                         T.CenterCrop(224),
    #                         T.ToTensor(),
    #                         normalize])
    
    if base_model == "resnet50":
        transforms = ResNet50_Weights.DEFAULT.transforms()
    elif base_model == "resnet18":
        transforms = ResNet18_Weights.DEFAULT.transforms()
    else:
        raise ValueError("Invalid base model")
    transforms = T.Compose([
        T.ToPILImage(), 
        # data augmentation
        T.AutoAugment(),
        transforms
    ])
                            


    train, val, test = [], [], []
    
    assert len(datasets) == len(drop), "Number of datasets and drop pairs should be the same"
    for i, dataset in enumerate(datasets):
        train_data, val_data, test_data = contrastive_datasets[dataset](transforms, dataset_dir=dataset_dir, use_complete=use_complete, drop=drop[i])
        train.append(train_data)
        val.append(val_data)
        test.append(test_data)
    
    
    train_data = combineDataSets(*train)
    val_data = combineDataSets(*val)
    test_data = combineDataSets(*test)

def get_dataLoaders(batch_size):
    """
    Get the data loaders for the datasets.

    Parameters:
        batch_size: int, batch size

    Returns:
        train_loader, val_loader, test_loader
    """
    global train_data, val_data, test_data

    assert train_data is not None, "Dataset not loaded"

    available_workers = math.ceil(os.cpu_count() / 2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=available_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=available_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, num_workers=available_workers)

    return train_loader, val_loader, test_loader

def get_datasetsMoco(datasets, batch_size, dataset_dir="datasets", base_model="resnet18"):
    """
    Prepares the datasets for contrastive training.

    Parameters:
        datasets: list, datasets to load
        batch_size: int, batch size
        dataset_dir: str, directory to save the datasets, default is 'datasets'
        base_model: str, base model, default is 'resnet18'
    """
    
    if base_model == "resnet50":
        transforms = ResNet50_Weights.DEFAULT.transforms()
    elif base_model == "resnet18":
        transforms = ResNet18_Weights.DEFAULT.transforms()
    else:
        raise ValueError("Invalid base model")
    transforms = T.Compose([
        T.ToPILImage(), 
        # data augmentation
        T.AutoAugment(),
        transforms
    ])

    train, val, test = [], [], []
    
    for i, dataset in enumerate(datasets):
        train_data, val_data, test_data = moco_datasets[dataset](transforms, dataset_dir=dataset_dir)
        train.append(train_data)
        val.append(val_data)
        test.append(test_data)
    
    
    train_data = combineDataSets(*train)
    val_data = combineDataSets(*val)
    test_data = combineDataSets(*test)

    assert train_data is not None, "Dataset not loaded"

    available_workers = math.ceil(os.cpu_count() / 2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=available_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False, num_workers=available_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, num_workers=available_workers)

    return train_loader, val_loader, test_loader
