import torch
from torch.optim import SGD
from torch.optim import AdamW
import numpy as np

def find_rotation_mat(points1, points2):
    """
    Function to find the rotation matrix between two sets of ordered 3D points.

    Parameters:
        points1: input containing a set of ordered 3D points
        points2: target containing the reference set of ordered 3D points

    Returns:
        transformation_matrix: torch.Tensor, 3x3 transformation matrix
    """

    # Calculate the covariance matrix 
    covariance_matrix = torch.mm(points1.t(), points2)

    # Calculate the singular value decomposition
    U, S, V = torch.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = torch.mm(V, U.t())

    # special reflection case
    if torch.det(rotation_matrix) < 0:
        U, S, V = torch.svd(covariance_matrix)
        V[2, :] *= -1
        rotation_matrix = torch.mm(V, U.t())

    return rotation_matrix

def find_scaling(points1, points2):
    """
    Function to find the scaling factor between two sets of centered and ordered 3D points.

    Parameters:
        points1: input containing a set of ordered 3D points
        points2: target containing the reference set of ordered 3D points

    Returns:
        scaling_factor: torch.Tensor, 1x1 scaling factor
    """
    scaling_factor = torch.norm(points1) / torch.norm(points2)

    return scaling_factor

# get loss for the whole batch
def get_loss(output, pose, weights=None, norm_factor=0.2, device='cuda'):
    """
    Calculate the loss between the predicted and target poses.

    Parameters:
        output: torch.Tensor, predicted pose
        pose: torch.Tensor, target pose
        weights: torch.Tensor, weights of the model for L2 normalization, default is None
        norm_factor: float, L2 normalization factor, default is 0.2
        device: str, device to run the model, default is 'cuda'

    Returns:
        loss: torch.Tensor, loss value
    """
    batch_size = output.shape[0]

    # vectors are a column vector and should be grouped by 3 (x, y, z)
    output = output.view(batch_size, -1, 3)
    pose = pose.view(batch_size, -1, 3)

    #print("output\n", output.shape)
    #print("pose\n", pose.shape)

    # center pose on mean point for each batch
    pose = pose - pose.mean(dim=1).unsqueeze(1)
    # center output on first point for each batch
    output = output - output.mean(dim=1).unsqueeze(1)

    #find rotation matrix for each batch
    batch_rotation_matrix = torch.zeros((batch_size, 3, 3)).to(device)
    # scaling_factor = torch.zeros((batch_size, 1)).to(device)

    with torch.no_grad():
        #print ("output before\n", output)
        for i in range(batch_size):
            #print(output[i])
            rotation_matrix = find_rotation_mat(pose[i], output[i])
            
            batch_rotation_matrix[i] = rotation_matrix

    output = torch.bmm(output, batch_rotation_matrix)
    
    output = output.view(-1, 1, 3)
    pose = pose.view(-1, 1, 3)
    
    # mean joint error
    loss = torch.mean(torch.cdist(output, pose, p=2))
    # breakpoint()

    # add L2 normalization factor for weights
    if weights is not None:
        weights = weights.view(-1)
        loss += norm_factor * torch.sum(weights**2)

    return loss

def get_optimizer(net, learning_rate, weight_decay):
    """
    Get the optimizer and scheduler for the network.

    Parameters:
        net: torch.nn.Module, network model
        learning_rate: float, learning rate
        weight_decay: float, weight decay

    Returns:
        optimizer: torch.optim, optimizer
        scheduler: torch.optim.lr_scheduler, scheduler
    """
    final_layer_weights = []
    rest_of_the_net_weights = []

    for name, param in net.named_parameters():
        #if from linear layer then add to final_layer_weights otherwise set requires_grad to false
        if 'fc' in name:
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)
            param.requires_grad = False
        
    #print (final_layer_weights)

    optimizer = AdamW(final_layer_weights, weight_decay=weight_decay, lr=learning_rate, eps=1e-6)
    #optimizer = SGD([ {'params': final_layer_weights, 'lr': learning_rate} ], weight_decay=weight_decay, momentum=momentum)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    return optimizer, scheduler

def project_points( points_3d, camera_parameters_str):
    """
    Project 3D points to 2D using camera parameters.

    Parameters:
        points_3d: numpy.array, 3D points
        camera_parameters_str: str, camera parameters

    Returns:
        points_2d: numpy.array, 2D points
    """
    # Load camera parameters
    camera_params = eval(camera_parameters_str)

    # Camera intrinsic matrix
    K = camera_params['K']

    # Rotation matrix and translation vector
    R = camera_params['R']
    t = camera_params['t']

    # Projection matrix
    P = np.dot(K, np.hstack((R, t)))

    # Homogeneous 3D points
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Project 3D points to 2D
    points_2d_homogeneous = np.dot(P, points_3d_homogeneous.T).T

    # Normalize homogeneous coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    # rseize to 128x128
    points_2d[:, 0] = points_2d[:, 0] * 128 / 1080
    points_2d[:, 1] = points_2d[:, 1] * 128 / 1080

    # center points, 2 is the center of the image
    points_2d[:, 0] = points_2d[:, 0] + 64 - points_2d[2, 0]
    points_2d[:, 1] = points_2d[:, 1] + 64 - points_2d[2, 1]

    return points_2d