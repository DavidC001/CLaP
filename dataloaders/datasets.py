import sys
sys.path.append(".")

from dataloaders.kinect import ContrastiveKinectDataset, ClusterKinectDataset, PoseKinectDataset
from dataloaders.skiPose import ContrastiveSkiDataset, ClusterSkiDataset, PoseSkiDataset
from dataloaders.panoptic import ContrastivePanopticDataset, ClusterPanopticDataset, PosePanopticDataset

import torch

class combineDataSets(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.length = sum(self.lengths)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError

contrastive_datasets = {'kinect': ContrastiveKinectDataset, 'skiPose': ContrastiveSkiDataset, 'panoptic': ContrastivePanopticDataset}
cluster_datasets = {'kinect': ClusterKinectDataset, 'skiPose': ClusterSkiDataset, 'panoptic': ClusterPanopticDataset}
pose_datasets = {'kinect': PoseKinectDataset, 'skiPose': PoseSkiDataset, 'panoptic': PosePanopticDataset}