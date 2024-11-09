import random
import re
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from contrastive_training.MoCo.model import get_moco_net
from contrastive_training.utils import get_datasetsMoco

from torch.utils.tensorboard import SummaryWriter

random.seed(42)

def get_loss(q, k_plus, k_negatives, t_plus, t_negative, batch_size):

    loss = 0

    num_view = int(k_plus.size(0) / batch_size)

    for j in range(batch_size):
      sum_plus = 0
      sum_negatives = 0

      for i in range(num_view):
          # Calculate the dot products
          q_dot_k_plus = torch.dot(q[j], k_plus[(i+j*num_view)]) / t_plus

          # Calculate the exponentials
          exp_q_dot_k_plus = torch.exp(q_dot_k_plus)

          sum_plus += exp_q_dot_k_plus

      for i in range(k_negatives.size(1)):
          # Calculate the dot products
          q_dot_k_negatives = torch.dot(q[j], k_negatives[:, i]) / t_negative

          # Calculate the exponentials
          exp_q_dot_k_negatives = torch.exp(q_dot_k_negatives)

          sum_negatives += exp_q_dot_k_negatives

      # Compute the loss
      loss += -torch.log(sum_plus / (sum_negatives + 1e-10))  # avoid division by zero

    return loss / batch_size

def get_optimizer(model, lr_encoder, lr_head, wd, momentum, epochs):
    final_layer_weights = []
    rest_of_the_net_weights = []

    for name, param in model.named_parameters():
        if name.startswith('fc'):
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    optimizer = torch.optim.SGD([
        {'params': rest_of_the_net_weights, 'lr': lr_encoder},
        {'params': final_layer_weights, 'lr': lr_head}
    ], weight_decay=wd, momentum=momentum)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    return optimizer, scheduler

def train_step(train_loader, model, optimizer, epoch, device, batch_size):

    # switch to train mode
    model.train()

    samples = 0.
    cumulative_loss = 0.

    for i, images in enumerate(train_loader):

        image_q = torch.tensor(images["query"]).to(device)

        images_k = []
        for img_k in images["keys"]:
          img_k = torch.tensor(img_k).to(device)
          images_k.append(img_k)


        # Forward pass to get embeddings
        embeddings_q, embeddings_k = model(image_q, images_k, device=device)

        # Extract positive and negative keys
        k_plus = embeddings_k
        k_negatives = model.module.queue.clone().detach()

        # Compute the loss
        loss = get_loss(embeddings_q, k_plus, k_negatives, model.module.T_plus, model.module.T_negative, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        samples += batch_size

        return cumulative_loss / samples

def val_step(val_loader, model, optimizer, epoch, device, batch_size):

    # switch to evaluate mode
    model.eval()

    samples = 0.
    cumulative_loss = 0.

    with torch.no_grad():
        #end = time.time()
        for i, images in enumerate(val_loader):
            image_q = torch.tensor(images["query"]).to(device)

            images_k = []
            for img_k in images["keys"]:
              img_k = torch.tensor(img_k).to(device)
              images_k.append(img_k)


            # Forward pass to get embeddings
            embeddings_q, embeddings_k = model(image_q, images_k, device=device)

            # Extract positive and negative keys
            k_plus = embeddings_k
            k_negatives = model.module.queue.clone().detach()

            # Compute the loss using get_loss function
            loss = get_loss(embeddings_q, k_plus, k_negatives, model.module.T_plus, model.module.T_negative, batch_size)

            cumulative_loss += loss.item()
            samples += batch_size

            return cumulative_loss / samples


def train_moco(model_dir, dataset_dir, datasets, save_every, batch_size=256, base_model='resnet18', name="moco", device='cuda', 
               learning_rate_encoder=0.03, learning_rate_head=0.1, momentum=0.9, weight_decay=0.0001,
               epochs=200, dim_out=128, K=65536, m=0.999, T_plus=0.07, T_negative=0.07, mode="multi", **others):

    if mode != "multi":
        raise ValueError("MoCo only supports multi mode")
    
    train_loader, val_loader, test_loader = get_datasetsMoco(datasets, batch_size, dataset_dir, base_model)

    # Initialize the model
    model = get_moco_net(base_model=base_model,dim_out=dim_out, K=K, m=m, T_plus=T_plus, T_negative=T_negative, device=device)
    model = model.to(device)

    # define optimizer
    optimizer, scheduler = get_optimizer(model, lr_encoder=learning_rate_encoder, lr_head=learning_rate_head, wd=weight_decay, momentum=momentum, epochs=epochs)

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
        model.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}.pt'.format(epoch)))
        #load optimizer
        optimizer.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}_optimizer.pt'.format(epoch)))
        #load scheduler
        scheduler.load_state_dict(torch.load(model_dir+ '/' + name+'/epoch_{:d}_scheduler.pt'.format(epoch)))

    cudnn.benchmark = True

    for e in range(epoch, epochs):
        # train for one epoch
        train_loss = train_step(train_loader, model, optimizer, e, device, batch_size)

        # evaluate on validation set
        val_loss = val_step(val_loader, model, optimizer, e, device, batch_size)

        scheduler.step()

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))

        writer.add_scalar("loss/train", train_loss, e+1) 
        writer.add_scalar("lr", scheduler.get_last_lr()[0], e+1) 
        writer.add_scalar("loss/val", val_loss, e+1)
        writer.flush()

        if (e+1) % save_every == 0:
            torch.save(model.module.encoder_q.state_dict(), model_dir+ '/'+name+'/epoch_{:d}.pt'.format(e+1))
            torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(e+1))
            torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(e+1))

    test_loss = val_step(test_loader, model, optimizer, epoch, device, batch_size)
    print('Test loss {:.5f}'.format(test_loss))
    writer.add_scalar("loss/test", test_loss, 0)
    
    writer.close()
        
    #save final model (if not saved already)
    if epochs % save_every != 0:
        torch.save(model.module.encoder_q.state_dict(), model_dir+ '/'+name+'/epoch_{:d}.pt'.format(epochs))
        torch.save(optimizer.state_dict(), model_dir+'/'+name+'/epoch_{:d}_optimizer.pt'.format(epochs))
        torch.save(scheduler.state_dict(), model_dir+'/'+name+'/epoch_{:d}_scheduler.pt'.format(epochs))

