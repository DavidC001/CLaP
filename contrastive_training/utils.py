import torch
import torch.nn as nn
import torchvision.transforms as T
from dataloaders.datasets import contrastive_datasets
from dataloaders.datasets import combineDataSets

#import RN50 transforms
from torchvision.models.resnet import  ResNet18_Weights, ResNet50_Weights

train_data, val_data, test_data = None, None, None

def load_datasets(datasets, dataset_dir="datasets", base_model="resnet18"):
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
    transforms = T.Compose([T.ToPILImage(), transforms])

    train, val, test = [], [], []
    
    for dataset in datasets:
        train_data, val_data, test_data = contrastive_datasets[dataset](transforms, dataset_dir=dataset_dir)
        train.append(train_data)
        val.append(val_data)
        test.append(test_data)
    
    
    train_data = combineDataSets(*train)
    val_data = combineDataSets(*val)
    test_data = combineDataSets(*test)

def get_dataLoaders(batch_size):
    global train_data, val_data, test_data

    assert train_data is not None, "Dataset not loaded"

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader