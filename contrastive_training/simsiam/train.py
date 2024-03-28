import sys
sys.path.append(".")

import os
import torch
import re

import torchvision.transforms as T

from contrastive_training.simsiam.model import get_siam_net

from dataloaders.datasets import contrastive_datasets

from torch.utils.tensorboard import SummaryWriter

def get_dataset(batch_size, dataset="panoptic"):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(128, 128)),
        ]
    )

    dataset = contrastive_datasets[dataset](transforms)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    return dataset, train_loader


from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_optimizer(model, lr, wd, momentum, epochs):

    optimizer = SGD([
        {'params': model.base.parameters(), 'fix_lr': False},
        {'params': model.predictor.parameters(), 'fix_lr': True}
    ], lr=lr, weight_decay=wd, momentum=momentum)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    return optimizer, scheduler

from torch import nn

def get_loss(p1, z2, p2, z1):

    def D(p, z):

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(p, z)

        return sim.mean()

    loss = - (1/2 * D(p1, z2) + 1/2 * D(p2, z1))

    return loss


from tqdm import tqdm

def train_step(net, data_loader, optimizer, cost_function, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    net.train()

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        image1 = batch['image1'].to(device)
        image2 = batch['image2'].to(device)

        x1, z1, p1 = net(image1)
        x2, z2, p2 = net(image2)

        loss = cost_function(p1, z2, p2, z1)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        cumulative_loss += loss.item()

        samples += image1.shape[0]

    return cumulative_loss / samples


def train_simsiam(model_dir="trained_models", name = "simsiam", batch_size=1024, device='cuda', learning_rate=0.01, weight_decay=0.000001, momentum=0.9, epochs=100, dataset="panoptic"):
    _, train_loader = get_dataset(batch_size, dataset)

    net = get_siam_net()
    net.to(device)

    optimizer, scheduler = get_optimizer(net, lr=learning_rate, wd=weight_decay, momentum=momentum, epochs=epochs)

    cost_function = get_loss

    writer = SummaryWriter(log_dir=model_dir+"/tensorboard/"+name)

    #create folder for model
    if not os.path.exists(model_dir+'/' + name):
        os.makedirs(model_dir+'/' + name)

    #get latest epoch
    epoch = 0
    for file in os.listdir(model_dir+'/' + name):
        if 'epoch' in file:
            e = int(re.findall(r'\d+', file)[0])
            if e > epoch:
                epoch = e

    print ('Starting training from epoch {:d}'.format(epoch))

    #load latest model
    if epoch > 0:
        net.load_state_dict(torch.load(model_dir+'/'+name+'/epoch_{:d}.pth'.format(epoch)))
        #load optimizer
        optimizer.load_state_dict(torch.load(model_dir+'/'+name+'/epoch_{:d}_optimizer.pth'.format(epoch)))
        #load scheduler
        scheduler.load_state_dict(torch.load(model_dir+'/'+name+'/epoch_{:d}_scheduler.pth'.format(epoch)))

    for e in range(epoch, epochs):
        train_loss = train_step(net, train_loader, optimizer, cost_function, device)

        scheduler.step()

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))

        writer.add_scalar("simclr/loss", train_loss, e+1) 
        writer.add_scalar("simclr/lr", scheduler.get_last_lr()[0], e+1) 
        writer.flush()

        if e % 20 == 0:
            torch.save(net.state_dict(), model_dir+'/'+name+'/epoch_{:d}.pth'.format(e+1))
            torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pth'.format(e+1))
            torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pth'.format(e+1))

        writer.close()

if __name__ == '__main__':
    train_simsiam(name = "simsiam", batch_size=180, epochs=100, learning_rate=0.3)
