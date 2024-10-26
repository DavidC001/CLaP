import sys
sys.path.append(".")

from dataloaders.ITOP import getContrastiveDatasetKinect, ClusterKinectDataset, getPoseDatasetKinect
from dataloaders.skiPose import getContrastiveDatasetSki, ClusterSkiDataset, getPoseDatasetSki
from dataloaders.panoptic import getContrastiveDatasetPanoptic, ClusterPanopticDataset, getPoseDatasetPanoptic

import torch

class combineDataSets(torch.utils.data.Dataset):
    """
    Combine multiple datasets into one dataset.
    """
    def __init__(self, *datasets):
        """
        Initialize the dataset.

        Parameters:
            *datasets: list of datasets
        """
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

contrastive_datasets = {'ITOP': getContrastiveDatasetKinect, 'skiPose': getContrastiveDatasetSki, 'panoptic': getContrastiveDatasetPanoptic}
cluster_datasets = {'ITOP': ClusterKinectDataset, 'skiPose': ClusterSkiDataset, 'panoptic': ClusterPanopticDataset}
pose_datasets = {'ITOP': getPoseDatasetKinect, 'skiPose': getPoseDatasetSki, 'panoptic': getPoseDatasetPanoptic}

out_joints = {'ITOP': 15, 'skiPose': 17, 'panoptic': 19}