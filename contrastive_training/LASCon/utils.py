import torch
import torch.nn as nn
import torchvision.transforms as T
from dataloaders.datasets import pose_datasets
from dataloaders.datasets import combineDataSets


def get_dataLoaders(datasets, batch_size, dataset_dir="datasets"):
    train_data, val_data, test_data = None, None, None
    
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(128, 128)),
        ]
    )

    train, val, test = [], [], []
    
    for dataset in datasets:
        train_data, val_data, test_data = pose_datasets[dataset](transforms, dataset_dir=dataset_dir)
        train.append(train_data)
        val.append(val_data)
        test.append(test_data)
    
    
    train_data = combineDataSets(*train)
    val_data = combineDataSets(*val)
    test_data = combineDataSets(*test)

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader