import sys
sys.path.append(".")

import os
import torch
import re
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn

from flash.core.optimizers import LARS

from tqdm import tqdm

#model is the same as simclr
from contrastive_training.simclr.model import get_simclr_net
from contrastive_training.LASCon.utils import get_dataLoaders

from pose_estimation.functions import find_rotation_mat, find_scaling

from torch.utils.tensorboard import SummaryWriter



def get_optimizer(model, lr, wd, momentum, epochs):
    final_layer_weights = []
    rest_of_the_net_weights = []

    for name, param in model.named_parameters():
        if name.startswith('fc'):
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    optimizer = LARS([
        {'params': rest_of_the_net_weights, 'lr': lr},
        {'params': final_layer_weights, 'lr': lr}
    ], weight_decay=wd, momentum=momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return optimizer, scheduler

def label_similarity(poses):
    batch_size = poses.size()[0]

    #calculate similarity between each pair of set of points of a pose, use low precision
    dist = torch.zeros((batch_size, batch_size), dtype=torch.int8).to(poses.device)
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            #rot_mat = find_rotation_mat(poses[i], poses[j])
            #poses_rot = torch.mm(poses[j], rot_mat)
            #scaling_factor = find_scaling(poses_rot, poses[i])
            #poses_rot = poses_rot * scaling_factor.item()
            #distance = torch.mean(torch.cdist(poses[i], poses[j], p=2))
            
            #cosine similarity between two poses
            distance = torch.nn.functional.cosine_similarity(poses[i][[16,8,14,5,11],:].view(-1), poses[j][[16,8,14,5,11],:].view(-1), dim=0)
            dist[i, j] = dist[j, i] = distance

    #normalized similarity
    #sim = torch.exp(-dist / torch.max(dist))
    
    return dist


def get_loss(emb, poses, t):
    batch_size = poses.size()[0]
    poses = poses.view(batch_size, -1, 3)

    #normalize to unit sphere
    emb = F.normalize(emb, p=2, dim=1)

    #calculate dot products
    sim = torch.mm(emb, emb.transpose(0,1))
    sim = torch.exp(sim / t)

    #calculate denominator
    denom = torch.sum(sim, dim=1) - torch.diagonal(sim, 0)

    #use LASCon loss
    lablesSim = label_similarity(poses)
    
    loss_vec = - torch.log(sim / denom) * lablesSim

    return loss_vec.sum() / emb.size()[0]


def train_step(net, data_loader, optimizer, cost_function, t, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    net.train()

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        images = batch['image'].to(device)
        poses = batch['poses_3d'].to(device)

        _, image_encoddings = net(images)

        loss = cost_function(image_encoddings, poses, t)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        cumulative_loss += loss.item()
        samples += images.shape[0]

    return cumulative_loss / samples

def val_step(net, data_loader, cost_function, t, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            images = batch['image'].to(device)
            poses = batch['poses_3d'].to(device)

            _, image_encoddings = net(images)

            loss = cost_function(image_encoddings, poses, t)

            cumulative_loss += loss.item()
            samples += images.shape[0]

    return cumulative_loss / samples


def train_LASCon(model_dir= "trained_models",name = "LASCon", dataset_dir="datasets", datasets=["panoptic"],
                  batch_size=1024, device='cuda', learning_rate=0.01, weight_decay=0.000001, momentum=0.9, t=0.6, epochs=100, save_every=10):
    
    
    train_loader, val_loader, test_loader = get_dataLoaders(batch_size=batch_size, datasets=datasets, dataset_dir=dataset_dir)

    net = get_simclr_net()
    net.to(device)

    optimizer, scheduler = get_optimizer(net, lr=learning_rate, wd=weight_decay, momentum=momentum, epochs=epochs)

    cost_function = get_loss

    writer = SummaryWriter(log_dir=model_dir+"/tensorboard/"+name)

    #create folder for model
    if not os.path.exists(model_dir+ '/' + name):
        os.makedirs(model_dir+ '/' + name)

    #get latest epoch
    epoch = 0
    for file in os.listdir(model_dir+ '/' + name):
        if 'epoch' in file:
            e = int(re.findall(r'\d+', file)[0])
            if e > epoch:
                epoch = e

    print ('Starting training from epoch {:d}'.format(epoch))

    #load latest model
    if epoch > 0:
        net.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}.pth'.format(epoch)))
        #load optimizer
        optimizer.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}_optimizer.pth'.format(epoch)))
        #load scheduler
        scheduler.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}_scheduler.pth'.format(epoch)))

    for e in range(epoch, epochs):
        train_loss = train_step(net, train_loader, optimizer, cost_function, t, device)
        val_loss = val_step(net, val_loader, cost_function, t, device)

        scheduler.step()

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))

        writer.add_scalar("loss/train", train_loss, e+1) 
        writer.add_scalar("lr", scheduler.get_last_lr()[0], e+1) 
        writer.add_scalar("loss/val", val_loss, e+1)
        writer.flush()

        if (e+1) % save_every == 0:
            torch.save(net.state_dict(), model_dir+ '/'+name+'/epoch_{:d}.pth'.format(e+1))
            torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pth'.format(e+1))
            torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pth'.format(e+1))

    #calculate test loss
    test_loss = val_step(net, test_loader, cost_function, t, device)
    print('Test loss {:.5f}'.format(test_loss))
    writer.add_scalar("loss/test", test_loss, 0)
    
    writer.close()
        
    #save final model (if not saved already)
    if epochs % save_every != 0:
        torch.save(net.state_dict(), model_dir+ '/'+name+'/epoch_{:d}.pth'.format(epochs))
        torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pth'.format(epochs))
        torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pth'.format(epochs))

if __name__ == "__main__":
    train_LASCon(name = "LASCon", datasets=["skiPose"],batch_size=650, epochs=100, learning_rate=0.3)
