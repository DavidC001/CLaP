import sys
sys.path.append(".")

from dataloaders.kinect import ContrastiveKinectDataset, ClusterKinectDataset, PoseKinectDataset
from dataloaders.skiPose import ContrastiveSkiDataset, ClusterSkiDataset, PoseSkiDataset
from dataloaders.panoptic import ContrastivePanopticDataset, ClusterPanopticDataset, PosePanopticDataset

contrastive_datasets = {'kinect': ContrastiveKinectDataset, 'skiPose': ContrastiveSkiDataset, 'panoptic': ContrastivePanopticDataset}
cluster_datasets = {'kinect': ClusterKinectDataset, 'skiPose': ClusterSkiDataset, 'panoptic': ClusterPanopticDataset}
pose_datasets = {'kinect': PoseKinectDataset, 'skiPose': PoseSkiDataset, 'panoptic': PosePanopticDataset}