import sys
sys.path.append(".")

from contrastive_training.LASCon.utils import get_dataLoaders

train, val, test = get_dataLoaders(["panoptic"], 1, dataset_dir="datasets")

#compute sim between all poses
import torch
import numpy as np
from tqdm import tqdm

from pose_estimation.functions import find_rotation_mat, find_scaling

def label_similarity(poses,name):
    batch_size = poses.size()[0]

    #calculate similarity between each pair of set of points of a pose, use low precision
    dist = torch.zeros((batch_size, batch_size))
    cosine = torch.ones((batch_size, batch_size))
    lin_cosine = torch.ones((batch_size, batch_size))
    for i in tqdm(range(batch_size)):
        for j in tqdm(range(i+1, batch_size), position=1):
            poses_i = poses[i].view(-1, 3)[[16,8,14,5,11],:]
            poses_j = poses[j].view(-1, 3)[[16,8,14,5,11],:]
            #center both poses
            poses_i = poses_i - torch.mean(poses_i, dim=0)
            poses_j = poses_j - torch.mean(poses_j, dim=0)

            rot_mat = find_rotation_mat(poses_i, poses_j)
            poses_rot = torch.mm(poses_i, rot_mat)
            scaling_factor = find_scaling(poses_i, poses_rot)
            pose_i_rot_scaled = poses_rot * scaling_factor.item()
            distance = torch.mean(torch.cdist(poses_j, pose_i_rot_scaled, p=2))
            # print("distance",distance.item())
            
            #cosine similarity between two poses
            cos = 0
            for joint in range(poses_i.shape[0]):
                cos += torch.nn.functional.cosine_similarity(poses_i[joint], poses_j[joint], dim=0)
            cos /= poses_i.shape[0]
            # print("cosine",cos.item())

            lin_cosine[i, j] = lin_cosine[j, i] = torch.nn.functional.cosine_similarity(poses_i.view(-1), poses_j.view(-1), dim=0)
            dist[i, j] = dist[j, i] = distance
            cosine[i, j] = cosine[j, i] = cos

    dist = torch.exp(-dist / torch.max(dist))
    from matplotlib import pyplot as plt
    #show side by side matrix
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(dist)
    axs[1].imshow(cosine)
    axs[2].imshow(lin_cosine)
    plt.show()


def compute_sim(dataloader, name="label_sim.pt"):
    poses = []
    i = 0
    for batch in tqdm(dataloader):
        poses.append(batch['poses_3d'].to("cuda"))
        i += 1
        if i == 10:
            break

    poses = torch.cat(poses, dim=0)
    print(poses.shape)

    sims = label_similarity(poses, name)

compute_sim(train, "train")
compute_sim(val, "val")
compute_sim(test, "test")