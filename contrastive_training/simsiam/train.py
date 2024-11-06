import sys
sys.path.append(".")

import os
import torch
import re

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from contrastive_training.simsiam.model import get_siam_net
from contrastive_training.utils import get_dataLoaders

from torch.utils.tensorboard import SummaryWriter

generator = torch.Generator().manual_seed(42)

def get_optimizer(model, lr_encoder, lr_head, wd, momentum, epochs):

    optimizer = SGD([
        {'params': model.base.parameters(), 'lr':lr_encoder},
        {'params': model.predictor.parameters(), 'lr':lr_head}
    ], weight_decay=wd, momentum=momentum)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return optimizer, scheduler


def get_loss(p1, z2, p2, z1):

    def D(p, z):

        cos = nn.CosineSimilarity(dim=1)
        sim = cos(p, z)

        return sim.mean()

    loss = - (1/2 * D(p1, z2) + 1/2 * D(p2, z1))

    return loss

def get_loss_multi(p, z):
    """
    Computes the SimSiam loss for multi-view data.

    Args:
        p (torch.Tensor): Predictor outputs of shape (N, K, D),
                          where N is the batch size,
                                K is the number of views,
                                D is the embedding dimension.
        z (torch.Tensor): Projector outputs of shape (N, K, D).
                          These are detached from the computation graph.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    N, K, D = p.shape  # Batch size, number of views, embedding dimension

    # Normalize the predictor and projector outputs
    p = F.normalize(p, dim=2)  # Shape: (N, K, D)
    z = F.normalize(z.detach(), dim=2)  # Shape: (N, K, D)

    # Compute cosine similarity between all pairs of predictor and projector outputs
    # Reshape p and z for batch matrix multiplication
    # p: (N, K, D) -> (N, K, D)
    # z: (N, K, D) -> (N, D, K)
    similarity = torch.bmm(p, z.transpose(1, 2))  # Shape: (N, K, K)

    # Create a mask to exclude self-pairs (i.e., when a view is paired with itself)
    # The mask has shape (N, K, K) with False on the diagonal and True elsewhere
    mask = ~torch.eye(K, device=p.device).bool().repeat(N, 1, 1)  # Shape: (N, K, K)

    # Apply the mask to filter out self-pairs
    # After masking, similarity contains similarities between distinct views within the same data point
    similarity = similarity[mask].view(N, K, K-1)  # Shape: (N, K, K-1)

    # Compute the SimSiam loss
    # For each predictor output p_i, compute its similarity with all z_j (j != i)
    # The loss is the negative mean similarity across all valid pairs
    loss = - similarity.mean()

    return loss

def train_step(net, data_loader, optimizer, cost_function, mode, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    net.train()

    for batch in tqdm(data_loader):

        if mode != 'multi':
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)

            x1, z1, p1 = net(image1)
            with torch.no_grad():
                x2, z2, p2 = net(image2)

            loss = cost_function(p1, z2, p2, z1)
        else:
            x, z, p = [], [], [] # [Cameras, Batch, Embedding]
            for images in batch['images']:
                images = images.to(device)
                x_, z_, p_ = net(images)
                x.append(x_) # [Batch, Embedding]
                z.append(z_)
                p.append(p_)
            # [Cameras, Batch, Embedding] -> [Batch, Cameras, Embedding]
            x = torch.stack(x, dim=1)
            z = torch.stack(z, dim=1)
            p = torch.stack(p, dim=1)
            loss = cost_function(p, z)


        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        cumulative_loss += loss.item()

        samples += 1

    return cumulative_loss / samples

def val_step(net, data_loader, cost_function, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    
    with torch.no_grad():
        for batch in tqdm(data_loader):

            if mode != 'multi':
                image1 = batch['image1'].to(device)
                image2 = batch['image2'].to(device)

                x1, z1, p1 = net(image1)
                x2, z2, p2 = net(image2)

                loss = cost_function(p1, z2, p2, z1)

            cumulative_loss += loss.item()

            samples += 1

    return cumulative_loss / samples


def train_simsiam(model_dir="trained_models", name = "simsiam",
                  batch_size=1024, device='cuda', 
                  learning_rate_encoder=0.01, learning_rate_head=0.1,
                  weight_decay=0.000001, momentum=0.9, epochs=100, 
                  save_every=10, base_model='resnet18', mode='simple',
                  **others):
    """
    Train SimSiam model
    
    Parameters:
        model_dir (str): Directory to save the model
        name (str): Name of the model
        batch_size (int): Batch size
        device (str): Device to run the model on
        learning_rate_encoder (float): Learning rate for the encoder
        learning_rate_head (float): Learning rate for the head
        weight_decay (float): Weight decay
        momentum (float): Momentum
        epochs (int): Number of epochs
        save_every (int): Save model every n epochs
        base_model (str): Base model to use for the SimSiam model
        mode (str): Mode of the dataset (simple, complete, multi)
    """
    
    train_loader, val_loader, test_loader = get_dataLoaders(batch_size)

    net = get_siam_net(base_model=base_model)
    net.to(device)

    optimizer, scheduler = get_optimizer(net.module, lr_encoder=learning_rate_encoder, lr_head=learning_rate_head, wd=weight_decay, momentum=momentum, epochs=epochs)

    cost_function = get_loss if mode != 'multi' else get_loss_multi

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
        net.load_state_dict(torch.load(model_dir+'/'+name+'/epoch_{:d}.pt'.format(epoch)))
        #load optimizer
        optimizer.load_state_dict(torch.load(model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(epoch)))
        #load scheduler
        scheduler.load_state_dict(torch.load(model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(epoch)))

    for e in range(epoch, epochs):
        train_loss = train_step(net, train_loader, optimizer, cost_function, mode, device)
        val_loss = val_step(net, val_loader, cost_function, mode, device)

        scheduler.step()

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))

        writer.add_scalar("contrastive/simsiam/loss_train", train_loss, e+1) 
        writer.add_scalar("contrastive/simsiam/lr_encoder", optimizer.param_groups[0]['lr'], e+1)
        writer.add_scalar("contrastive/simsiam/lr_head", optimizer.param_groups[1]['lr'], e+1)
        writer.add_scalar("contrastive/simsiam/loss_val", val_loss, e+1)
        writer.flush()

        if (e+1) % save_every == 0:
            torch.save(net.state_dict(), model_dir+'/'+name+'/epoch_{:d}.pt'.format(e+1))
            torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(e+1))
            torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(e+1))

    #calculate test loss
    test_loss = val_step(net, test_loader, cost_function, device)
    print('Test loss {:.5f}'.format(test_loss))
    writer.add_scalar("contrastive/simsiam/loss_test", test_loss, 0)

    writer.close()
    
    # save the final model (if not saved already)
    if epochs % save_every != 0:
        torch.save(net.state_dict(), model_dir+'/'+name+'/epoch_{:d}.pt'.format(epochs))
        torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(epochs))
        torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(epochs))

if __name__ == '__main__':
    train_simsiam(name = "simsiam", batch_size=180, epochs=100, learning_rate=0.3)
