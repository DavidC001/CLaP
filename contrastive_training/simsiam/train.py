import sys
sys.path.append(".")

import os
import torch
import re

import torchvision.transforms as T
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from tqdm import tqdm

from contrastive_training.simsiam.model import get_siam_net

from dataloaders.datasets import contrastive_datasets
from dataloaders.datasets import combineDataSets

from torch.utils.tensorboard import SummaryWriter

generator = torch.Generator().manual_seed(42)

def get_dataset(batch_size, datasets=["panoptic"], dataset_dir="datasets"):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(128, 128)),
        ]
    )

    data = combineDataSets(*[contrastive_datasets[dataset](transforms, dataset_dir=dataset_dir) for dataset in datasets])

    num_samples = len(data)

    training_samples = int(num_samples * 0.8 + 1)
    val_samples = num_samples - training_samples

    training_data, val_data, = torch.utils.data.random_split(
        data, [training_samples, val_samples], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=False)

    return training_data, val_data, train_loader, val_loader


def get_optimizer(model, lr, wd, momentum, epochs):

    optimizer = SGD([
        {'params': model.base.parameters()},
        {'params': model.predictor.parameters()}
    ], lr=lr, weight_decay=wd, momentum=momentum)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return optimizer, scheduler


def get_loss(p1, z2, p2, z1):

    def D(p, z):

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(p, z)

        return sim.mean()

    loss = - (1/2 * D(p1, z2) + 1/2 * D(p2, z1))

    return loss


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

def val_step(net, data_loader, cost_function, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    net.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        image1 = batch['image1'].to(device)
        image2 = batch['image2'].to(device)

        x1, z1, p1 = net(image1)
        x2, z2, p2 = net(image2)

        loss = cost_function(p1, z2, p2, z1)

        cumulative_loss += loss.item()

        samples += image1.shape[0]

    return cumulative_loss / samples


def train_simsiam(model_dir="trained_models", name = "simsiam",  dataset_dir="datasets", datasets="panoptic",
                  batch_size=1024, device='cuda', learning_rate=0.01, weight_decay=0.000001, momentum=0.9, epochs=100, save_every=10):
    
    _, _, train_loader, val_loader = get_dataset(batch_size, datasets, dataset_dir)

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
        val_loss = val_step(net, val_loader, cost_function, device)

        scheduler.step()

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))

        writer.add_scalar(name+"/train_loss", train_loss, e+1) 
        writer.add_scalar(name+"/lr", scheduler.get_last_lr()[0], e+1) 
        writer.add_scalar(name+"/val_loss", val_loss, e+1)
        writer.flush()

        if (e+1) % save_every == 0:
            torch.save(net.state_dict(), model_dir+'/'+name+'/epoch_{:d}.pth'.format(e+1))
            torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pth'.format(e+1))
            torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pth'.format(e+1))

        writer.close()
    
    # save the final model (if not saved already)
    if epochs % save_every != 0:
        torch.save(net.state_dict(), model_dir+'/'+name+'/epoch_{:d}.pth'.format(epochs))
        torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pth'.format(epochs))
        torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pth'.format(epochs))

if __name__ == '__main__':
    train_simsiam(name = "simsiam", batch_size=180, epochs=100, learning_rate=0.3)
