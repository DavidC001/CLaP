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



def get_optimizer(model, lr_encoder, lr_head, wd, momentum, epochs):
    final_layer_weights = []
    rest_of_the_net_weights = []

    for name, param in model.named_parameters():
        if name.startswith('fc'):
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    optimizer = LARS([
        {'params': rest_of_the_net_weights, 'lr': lr_encoder},
        {'params': final_layer_weights, 'lr': lr_head}
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

def get_loss_multi(encodings, temperature):
    """
    Computes the multi-view contrastive loss.

    Args:
        encodings (torch.Tensor): Tensor of shape (N, K, D), where
            N = batch size,
            K = number of views per data point,
            D = encoding dimension.
        temperature (float): Temperature parameter for scaling similarities.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    N, K, D = encodings.shape  # Batch size, number of views, embedding dimension

    # Normalize the encodings
    encodings = F.normalize(encodings, p=2, dim=2)  # Shape: (N, K, D)

    # Reshape to (N*K, D) for similarity computation
    encodings = encodings.view(N * K, D)  # Shape: (N*K, D)

    # Compute similarity matrix (cosine similarity)
    similarity_matrix = torch.matmul(encodings, encodings.T)  # Shape: (N*K, N*K)

    # Scale similarities by temperature and exponentiate
    similarity_matrix = torch.exp(similarity_matrix / temperature)  # Shape: (N*K, N*K)

    # Create labels for identifying positive pairs
    labels = torch.arange(N).repeat_interleave(K).to(encodings.device)  # Shape: (N*K,)
    labels = labels.view(N, K)  # Shape: (N, K)
    labels = labels.view(N * K, 1)  # Shape: (N*K, 1)

    # Compare labels to create mask of positives
    positives_mask = torch.eq(labels, labels.T).float()  # Shape: (N*K, N*K)

    # Remove self-similarity by setting diagonal to 0
    positives_mask = positives_mask - torch.eye(N * K).to(encodings.device)

    # Compute numerator: sum of similarities with positives
    numerator = torch.sum(similarity_matrix * positives_mask, dim=1)  # Shape: (N*K,)

    # Compute denominator: sum of similarities with all except itself
    denominator = torch.sum(similarity_matrix, dim=1) - torch.diagonal(similarity_matrix)  # Shape: (N*K,)

    # To avoid division by zero
    denominator = torch.clamp(denominator, min=1e-8)

    # Compute loss for each anchor
    loss = -torch.log(numerator / denominator)

    # Average loss over all anchors
    loss = loss.mean()

    return loss


def train_step(net, data_loader, optimizer, cost_function, t, mode, device='cuda'):

    samples = 0.
    cumulative_loss = 0.
    net.train()

    for batch in tqdm(data_loader):

        if mode != "multi":
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)

            _, image1_encoddings = net(image1)
            _, image2_encoddings = net(image2)

            loss = cost_function(image1_encoddings, image2_encoddings, t)
        else:
            encodings = []
            for images in batch["images"]:
                images = images.to(device)
                _, encoding = net(images)
                encodings.append(encoding)
            encodings = torch.stack(encodings, dim=1)
            loss = cost_function(encodings, t)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        cumulative_loss += loss.item()
        samples += 1

    return cumulative_loss / samples

def val_step(net, data_loader, cost_function, t, mode, device='cuda'):
    samples = 0.
    cumulative_loss = 0.
    with torch.no_grad():
        for batch in tqdm(data_loader):

            if mode != "multi":
                image1 = batch['image1'].to(device)
                image2 = batch['image2'].to(device)

                _, image1_encoddings = net(image1)
                _, image2_encoddings = net(image2)

                loss = cost_function(image1_encoddings, image2_encoddings, t)
            else:
                encodings = []
                for images in batch["images"]:
                    images = images.to(device)
                    _, encoding = net(images)
                    encodings.append(encoding)
                encodings = torch.stack(encodings, dim=1)
                loss = cost_function(encodings, t)

            cumulative_loss += loss.item()
            samples += 1

    return cumulative_loss / samples


def train_simclr(model_dir= "trained_models",name = "simclr",
                  batch_size=1024, device='cuda', 
                  learning_rate_encoder=0.01, learning_rate_head=0.1,
                  weight_decay=0.000001, momentum=0.9, temperature=0.6, 
                  epochs=100, save_every=10, base_model='resnet18', 
                  mode='simple',
                  **others):
    """
    Train a SimCLR model

    Parameters
        model_dir (str): directory to save the model
        name (str): name of the model
        batch_size (int): batch size
        device (str): device to run the model on
        learning_rate_encoder (float): learning rate for the resnet encoder
        learning_rate_head (float): learning rate for the head
        weight_decay (float): weight decay
        momentum (float): momentum
        temperature (float): temperature for the contrastive loss
        epochs (int): number of epochs
        save_every (int): save model every n epochs
        base_model (str): base model to use for the encoder (resnet18 or resnet50)
        mode (str): mode of the dataset (simple, complete, multi)
    """
    
    train_loader, val_loader, test_loader = get_dataLoaders(batch_size)

    net = get_simclr_net(base_model=base_model)
    net.to(device)

    optimizer, scheduler = get_optimizer(net, lr_encoder=learning_rate_encoder, lr_head=learning_rate_head, wd=weight_decay, momentum=momentum, epochs=epochs)

    cost_function = get_loss if mode != "multi" else get_loss_multi

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
        net.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}.pt'.format(epoch)))
        #load optimizer
        optimizer.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}_optimizer.pt'.format(epoch)))
        #load scheduler
        scheduler.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}_scheduler.pt'.format(epoch)))

    for e in range(epoch, epochs):
        train_loss = train_step(net, train_loader, optimizer, cost_function, temperature, mode, device)
        val_loss = val_step(net, val_loader, cost_function, temperature, mode, device)

        scheduler.step()

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))

        writer.add_scalar("contrastive/simclr/loss_train", train_loss, e+1) 
        writer.add_scalar("contrastive/simclr/lr_encoder", optimizer.param_groups[0]['lr'], e+1)
        writer.add_scalar("contrastive/simclr/lr_head", optimizer.param_groups[1]['lr'], e+1)
        writer.add_scalar("contrastive/simclr/loss_val", val_loss, e+1)
        writer.flush()

        if (e+1) % save_every == 0:
            torch.save(net.state_dict(), model_dir+ '/'+name+'/epoch_{:d}.pt'.format(e+1))
            torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(e+1))
            torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(e+1))

    #calculate test loss
    test_loss = val_step(net, test_loader, cost_function, temperature, device)
    print('Test loss {:.5f}'.format(test_loss))
    writer.add_scalar("contrastive/simclr/loss_test", test_loss, 0)
    
    writer.close()
        
    #save final model (if not saved already)
    if epochs % save_every != 0:
        torch.save(net.state_dict(), model_dir+ '/'+name+'/epoch_{:d}.pt'.format(epochs))
        torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(epochs))
        torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(epochs))

if __name__ == "__main__":
    train_simclr(name = "simclr", batch_size=180, epochs=100, learning_rate=0.3)
