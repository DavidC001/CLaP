import sys
sys.path.append('.')
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import re
from pose_estimation.functions import get_loss



def training_step(net, data_loader, optimizer, cost_function, device='cuda'):
    batches = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    samples = 0.0

    net.train()

    for batch_idx, batch in enumerate(tqdm(data_loader)):

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

        samples += images.shape[0]

        

    return cumulative_loss / samples


def test_step(net, data_loader, cost_function, device='cuda'):
    batches = 0.
    cumulative_loss = 0.
    cumulative_accuracy = 0.
    samples = 0.

    net.eval()

    with torch.no_grad():
        #show image and poses
        from matplotlib import pyplot as plt
        import cv2

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images = batch['image']
            poses = batch['poses_3d']
            # cv2.imshow("image", images[0].cpu().numpy().transpose(1,2,0))

            images = images.to(device)
            poses = poses.to(device)

            output = net(images)

            pose = poses[0].view(-1,3).cpu()

            loss = cost_function(output, poses, device=device)
            cumulative_loss += loss.item()

            pose = poses[0].view(-1,3).cpu()

            samples += images.shape[0]

    return cumulative_loss / samples

def train (model, optimizer, scheduler, train_loader, val_loader, test_loader, epochs, save_every=10, device='cuda', model_dir="trained_models", name="model"):
    net = model
    epoch = 0
    cost_function = get_loss

    #redo this
    info_file = os.path.join(model_dir, name, "info.txt").replace("\\", "/")
    model_file = os.path.join(model_dir, name, "epoch_").replace("\\", "/")
    model_dir = os.path.join(model_dir, name).replace("\\", "/")
    optimizer_file = os.path.join(model_dir, "optimizer_epoch").replace("\\", "/")
    scheduler_file = os.path.join(model_dir, "scheduler_epoch").replace("\\", "/")
        
    tensorboard_tag = name

    writer = SummaryWriter(log_dir=model_dir, filename_suffix="_"+tensorboard_tag)


    #load weights
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        #remove non weight files
        files = [file for file in files if '.pt' in file]
        if len(files) > 1:
            epoch = max([int(re.findall(r'\d+', file)[0]) for file in files])
            net.load_state_dict(torch.load(model_file+str(epoch)+'.pt', map_location=torch.device('cuda')))
            print("Loaded weights from epoch", epoch)
        else:
            print("No weights found")
            f = open(info_file, "w")
            f.close()

        
    if os.path.exists(optimizer_file+str(epoch)+'.pt'):
        print("Loaded optimizer from epoch", epoch)
        optimizer.load_state_dict(torch.load(optimizer_file+str(epoch)+'.pt'))
        scheduler.load_state_dict(torch.load(scheduler_file+str(epoch)+'.pt'))


    for e in range(epoch, epochs):

        train_loss = training_step(net, train_loader, optimizer, cost_function, device)
        val_loss = test_step(net, val_loader, cost_function, device)

        scheduler.step()

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

        writer.add_scalar(tensorboard_tag+'/Loss/train', train_loss, e+1)
        writer.add_scalar(tensorboard_tag+'/Loss/val', val_loss, e+1)
        writer.add_scalar(tensorboard_tag+'/lr', scheduler.get_last_lr()[0], e+1)
        writer.flush()

    if epochs % save_every != 0:
        torch.save(net.state_dict(), model_file+str(epochs)+'.pt')
        torch.save(optimizer.state_dict(), optimizer_file+str(epochs)+'.pt')
        torch.save(scheduler.state_dict(), scheduler_file+str(epochs)+'.pt')

    print('After training:')
    train_loss, train_accuracy = test_step(net, train_loader, cost_function, device)
    val_loss, val_accuracy = test_step(net, val_loader, cost_function, device)
    test_loss, test_accuracy = test_step(net, test_loader, cost_function, device)

    print('\tTraining loss {:.5f}, Training Acc {:.4f}'.format(train_loss, train_accuracy))
    print('\tValidation loss {:.5f}, Validation Acc {:.4f}'.format(val_loss, val_accuracy))
    print('\tTest loss {:.5f}, Test Acc {:.4f}'.format(test_loss, test_accuracy))
    print('-----------------------------------------------------')

    #write information to file
    f = open(info_file, "a")
    f.write('After training:\n')
    f.write('\tTraining loss {:.5f}, Training Acc {:.4f}\n'.format(train_loss, train_accuracy))
    f.write('\tValidation loss {:.5f}, Validation Acc {:.4f}\n'.format(val_loss, val_accuracy))
    f.write('\tTest loss {:.5f}, Test Acc {:.4f}\n'.format(test_loss, test_accuracy))
    f.write('-----------------------------------------------------\n')
    f.close()

    writer.close()
