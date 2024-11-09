import sys
sys.path.append('.')
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import re
from pose_estimation.functions import get_loss
from copy import deepcopy


def training_step(net, data_loader, optimizer, cost_function, device='cuda'):
    """
    Make a training step for the pose estimation model.

    Parameters:
        net: torch.nn.Module, network model
        data_loader: torch.utils.data.DataLoader, data loader
        optimizer: torch.optim.Optimizer, optimizer
        cost_function: function, cost function
        device: str, device, default is 'cuda'

    Returns:
        loss: float, mean loss of the training step
    """
    cumulative_loss = 0.0
    samples = 0.0

    net.train()

    for batch in tqdm(data_loader):

        images = batch['image']
        poses = batch['poses_3d']

        images = images.to(device)
        poses = poses.to(device)

        output = net(images)

        loss = cost_function(output, poses, device=device)
        cumulative_loss += loss.item()

        loss.backward()
        # print(loss.item())

        optimizer.step()

        optimizer.zero_grad()

        samples += 1

        

    return cumulative_loss / samples


def test_step(net, data_loader, cost_function, device='cuda'):
    """
    Make a test step for the pose estimation model.

    Parameters:
        net: torch.nn.Module, network model
        data_loader: torch.utils.data.DataLoader, data loader
        cost_function: function, cost function
        device: str, device, default is 'cuda'

    Returns:
        loss: float, mean loss of the test step
    """
    cumulative_loss = 0.
    samples = 0.

    net.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch['image']
            poses = batch['poses_3d']

            images = images.to(device)
            poses = poses.to(device)

            output = net(images)

            loss = cost_function(output, poses, device=device)
            cumulative_loss += loss.item()

            samples += 1

    return cumulative_loss / samples


def train (model, optimizer, scheduler, train_loader, val_loader, test_loader, epochs, save_every=10, device='cuda', model_dir="trained_models", name="model", patience = 2):
    """
    Train the pose estimation model.

    Parameters:
        model: torch.nn.Module, network model
        optimizer: torch.optim.Optimizer, optimizer
        scheduler: torch.optim.lr_scheduler, scheduler
        train_loader: torch.utils.data.DataLoader, training data loader
        val_loader: torch.utils.data.DataLoader, validation data loader
        test_loader: torch.utils.data.DataLoader, test data loader
        epochs: int, number of epochs
        save_every: int, save every n epochs, default is 10
        device: str, device, default is 'cuda'
        model_dir: str, directory to save the trained models, default is 'trained_models'
        name: str, model name, default is 'model'
        patience: int, patience for early stopping, default is 2
    
    """
    net = model
    epoch = 0
    cost_function = get_loss

    tensorboard_tag = "estimator"
    tensorboard_dir = os.path.join(model_dir, "tensorboard", name).replace("\\", "/")
    
    info_file = os.path.join(model_dir, name, "info.txt").replace("\\", "/")
    model_file = os.path.join(model_dir, name, "epoch_").replace("\\", "/")
    model_dir = os.path.join(model_dir, name).replace("\\", "/")
    optimizer_file = os.path.join(model_dir, "optimizer_epoch").replace("\\", "/")
    scheduler_file = os.path.join(model_dir, "scheduler_epoch").replace("\\", "/")

    #create folder for model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir, filename_suffix="_"+name)

    #load weights
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        #remove non weight files
        files = [file for file in files if '.pt' in file]
        if len(files) > 1:
            epoch = max([int(re.findall(r'\d+', file)[0]) for file in files])
            net.load_state_dict(torch.load(model_file+str(epoch)+'.pt', map_location=torch.device('cuda')))
            print("\tLoaded weights from epoch", epoch)
        else:
            print("\tNo weights found")
            f = open(info_file, "w")
            f.close()

        
    if os.path.exists(optimizer_file+str(epoch)+'.pt'):
        optimizer.load_state_dict(torch.load(optimizer_file+str(epoch)+'.pt'))
        print("\tLoaded optimizer from epoch", epoch)
        scheduler.load_state_dict(torch.load(scheduler_file+str(epoch)+'.pt'))
        print("\tLoaded scheduler from epoch", epoch)

    print("\tStarting Training:")
    patience_counter = 0
    min_val_loss = 1000000
    best_model = None
    
    for e in tqdm(range(epoch, epochs)):

        train_loss = training_step(net, train_loader, optimizer, cost_function, device)
        val_loss = test_step(net, val_loader, cost_function, device)

        scheduler.step(val_loss)

        print('Epoch: {:d}'.format(e+1))
        print('\tTraining loss {:.5f}'.format(train_loss))
        print('\tValidation loss {:.5f}'.format(val_loss))
        print('-----------------------------------------------------')

        if (e+1) % save_every == 0:
            torch.save(net.state_dict(), model_file+str(e+1)+'.pt')
            torch.save(optimizer.state_dict(), optimizer_file+str(e+1)+'.pt')
            torch.save(scheduler.state_dict(), scheduler_file+str(e+1)+'.pt')
        
        #write information to file
        f = open(info_file, "a")
        f.write('Epoch: {:d}\n'.format(e+1))
        f.write('\tTraining loss {:.5f}\n'.format(train_loss))
        f.write('\tValidation loss {:.5f}\n'.format(val_loss))
        f.write('-----------------------------------------------------\n')
        f.close()

        writer.add_scalar(tensorboard_tag+'/Loss/train', train_loss*1000, e+1)
        writer.add_scalar(tensorboard_tag+'/Loss/val', val_loss*1000, e+1)
        writer.add_scalar(tensorboard_tag+'/lr', optimizer.param_groups[0]["lr"] , e+1)
        writer.flush()

        if e > 0:
            if val_loss > min_val_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                min_val_loss = val_loss
                best_model = deepcopy(net)
        
        if patience_counter > patience:
            print("Early stopping")
            break
    
    if (best_model):
        torch.save(best_model.state_dict(), model_file+str(e+1)+'.pt')
    else:
        best_model = net

    print('After training:')
    train_loss = test_step(best_model, train_loader, cost_function, device)
    val_loss = test_step(best_model, val_loader, cost_function, device)
    test_loss = test_step(best_model, test_loader, cost_function, device)

    print('\tTraining loss {:.5f}'.format(train_loss*1000))
    print('\tValidation loss {:.5f}'.format(val_loss*1000))
    print('\tTest loss {:.5f}'.format(test_loss*1000))

    #write information to file
    f = open(info_file, "a")
    f.write('After training:\n')
    f.write('\tTraining loss {:.5f}\n'.format(train_loss*1000))
    f.write('\tValidation loss {:.5f}\n'.format(val_loss*1000))
    f.write('\tTest loss {:.5f}\n'.format(test_loss*1000))
    f.write('-----------------------------------------------------\n')
    f.close()

    
    writer.add_scalar(tensorboard_tag+'/final_train_loss', train_loss*1000, 0)
    writer.add_scalar(tensorboard_tag+'/final_val_loss', val_loss*1000, 0)
    writer.add_scalar(tensorboard_tag+'/final_test_loss', test_loss*1000, 0)
    writer.flush()

    writer.close()
