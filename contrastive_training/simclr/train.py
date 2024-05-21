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

from contrastive_training.simclr.model import get_simclr_net
from contrastive_training.utils import get_dataLoaders

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


def get_loss(geom_encoddings, app_encoddings, t):
    geom_encoddings = F.normalize(geom_encoddings, p=2, dim=1)
    app_encoddings = F.normalize(app_encoddings, p=2, dim=1)

    def get_sim(zi, zj, t):
        cosi = torch.nn.CosineSimilarity(dim=1)
        return torch.exp(cosi(zi, zj) / t)

    num = get_sim(geom_encoddings, app_encoddings, t)
    num = torch.cat([num, num])

    batch = torch.cat([geom_encoddings, app_encoddings])
    batch = batch / batch.norm(dim=1)[:, None]
    sim = torch.mm(batch, batch.transpose(0,1))
    sim = torch.exp(sim / t)

    denom = torch.sum(sim, dim=1) - torch.diagonal(sim, 0)
    loss_vec = - torch.log(num / denom)

    loss = loss_vec.sum() / batch.size()[0]

    return loss


def train_step(net, data_loader, optimizer, cost_function, t, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    net.train()

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        image1 = batch['image1'].to(device)
        image2 = batch['image2'].to(device)

        _, image1_encoddings = net(image1)
        _, image2_encoddings = net(image2)

        loss = cost_function(image1_encoddings, image2_encoddings, t)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        cumulative_loss += loss.item()
        samples += image1.shape[0]

    return cumulative_loss / samples

def val_step(net, data_loader, cost_function, t, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)

            _, image1_encoddings = net(image1)
            _, image2_encoddings = net(image2)

            loss = cost_function(image1_encoddings, image2_encoddings, t)

            cumulative_loss += loss.item()
            samples += image1.shape[0]

    return cumulative_loss / samples


def train_simclr(model_dir= "trained_models",name = "simclr", dataset_dir="datasets", datasets=["panoptic"],
                  batch_size=1024, device='cuda', learning_rate=0.01, weight_decay=0.000001, momentum=0.9, t=0.6, epochs=100, save_every=10, base_model='resnet18'):
    
    
    train_loader, val_loader, test_loader = get_dataLoaders(batch_size)

    net = get_simclr_net(base_model=base_model)
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
    train_simclr(name = "simclr", batch_size=180, epochs=100, learning_rate=0.3)
