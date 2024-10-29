import re
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch
import os
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


connections = [(7,8),(6,7),(2,6),(2,12),(12,13),(13,14),(2,0),(0,3),(0,9),(3,4),(4,5),(9,10),(10,11),(0,16)]


'''
Define the SimCLR base model with encoder and projection head
'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return x, self.layers(x)
    
from torchvision.models import resnet50, ResNet50_Weights


def get_simclr_net():
    """
    Returns the SimCLR network model.

    This function creates a SimCLR network model by using a pre-trained ResNet50 model
    as the backbone and replacing the fully connected layer with a custom MLP layer.

    Returns:
        model (nn.Module): SimCLR network model.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = MLP(2048, 2048, 128)

    return model

'''
Define the SimSiam projector and predictor
'''
class Projector(nn.Module):
    def __init__(self, input_dim, out_dim, hidder_proj):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidder_proj),
            nn.BatchNorm1d(hidder_proj),
            nn.ReLU(inplace=True),

            nn.Linear(hidder_proj, hidder_proj),
            nn.BatchNorm1d(hidder_proj),
            nn.ReLU(inplace=True),

            nn.Linear(hidder_proj, out_dim),
        )

    def forward(self, x):
        return x, self.proj(x)

class SiamMLP(nn.Module):
    def __init__(self, base, input_dim, out_dim, hidder_proj, hidden_pred):
        super().__init__()

        self.base = base

        base.fc = Projector(input_dim, out_dim, hidder_proj)

        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_pred),
            nn.BatchNorm1d(hidden_pred),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_pred, out_dim)
        )


    def forward(self, x):
        x, projections =  self.base(x)

        predictions = self.predictor(projections)

        return x, projections.detach(), predictions


'''
    Define the linear regression last layers for the pose estimation task
'''
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        #initialize weights and biases to 0
        self.layers = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 18*3))

    def forward(self, x):
        z = self.layers(x)
        return z


def get_linear_evaluation_model(path, base, siam=True):
    """
    Returns a linear evaluation model based on the given path and base model.

    Args:
        path (str): The path to the saved model state dictionary.
        base (torch.nn.Module): The base model.
        siam (bool, optional): Whether the base model is SimSiam network. Defaults to True.

    Returns:
        torch.nn.Module: The linear evaluation model.
    """

    base.load_state_dict(torch.load(path, map_location=torch.device('cuda')))

    if siam:
        base = base.base
        base.fc = Linear()
    else:
        base.fc = Linear()

    return base

#load simclr model
simclr_path = 'trained_models/ver1.pt'
simclr = get_simclr_net()

#get last model epoch in trained_models/sim/ folder

#get list of files in directory
files = os.listdir("trained_models/sim_2layer")
files = [f for f in files if re.match(r'sim_linear_epoch\d+.pt', f)]

#sort files by epoch number
files.sort(key=lambda f: int(re.sub('\D', '', f)))
#get last file
path = "trained_models/sim_2layer/" + files[-1]

print("Loading model from: " + path)

#load model

simclr_model = get_linear_evaluation_model(simclr_path, simclr, False)
simclr_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

simclr_model.eval()

# OpenCV setup for webcam capture
cap = cv2.VideoCapture(0)

# Define a transformation to preprocess the webcam image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



def get_simclr_predictions(frame):
    """
    Applies the SimCLR model to a given frame and returns the predicted coordinates.

    Args:
        frame (numpy.ndarray): The input frame.

    Returns:
        torch.Tensor: The predicted coordinates after applying SimCLR preprocessing and transformations.
    """
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the SimCLR preprocessing transform
    input_image = transform(frame_rgb).unsqueeze(0)

    # Forward pass through the SimCLR model
    with torch.no_grad():
        pred_coords = simclr_model(input_image)[0].reshape((-1,3)).detach()
        #add first point to output (0,0,0)
        pred_coords = torch.cat((torch.zeros((1, 3)), pred_coords), 0)

        #center on point 2
        pred_coords = pred_coords - pred_coords[2]

        spine = pred_coords[0] - pred_coords[2]
        unit_spine = spine / torch.norm(spine)

        unit_vector_y = torch.tensor([0,1,0], dtype=torch.float32)

        v = torch.cross(unit_spine, unit_vector_y)
        s = torch.norm(v)
        c = torch.dot(unit_spine, unit_vector_y)

        skew_symmetric_cross_product_matrix = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float32)
        rotation_matrix = torch.eye(3, dtype=torch.float32) + skew_symmetric_cross_product_matrix + torch.matmul(skew_symmetric_cross_product_matrix, skew_symmetric_cross_product_matrix) * ((1 - c) / (s ** 2))

        # rotate points
        pred_coords = torch.matmul(rotation_matrix, pred_coords.T).T

        #flip y if spine is pointing down
        if pred_coords[0][1] < pred_coords[2][1]:
            pred_coords[:,1] = -pred_coords[:,1]

        #scale to be in range [-100, 100]
        pred_coords = pred_coords * 100 / torch.max(torch.abs(pred_coords))
        
    return pred_coords

def update(num, scatter, lines, texts):
    """
    Update function for live_demo.

    Args:
        num (int): The frame number.
        scatter (matplotlib.collections.PathCollection): The scatter plot object.
        lines (List[matplotlib.lines.Line3D]): The list of line objects.
        texts (List[matplotlib.text.Text]): The list of text objects.
    """
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get SimCLR predictions for the current frame
    predictions = get_simclr_predictions(frame)

    # Display the frame with SimCLR predictions
    cv2.imshow('frame', frame)

    # show 3d points in matplotlib
    scatter._offsets3d = (predictions[:,0], predictions[:,2], predictions[:,1])

    # draw numbers
    for i, text in enumerate(texts):
        text.set_position((predictions[i][0], predictions[i][2], predictions[i][1]))
    
    # draw lines
    for i, line in enumerate(lines):
        start_point = predictions[connections[i][0]]
        end_point = predictions[connections[i][1]]
        line.set_data_3d([start_point[0], end_point[0]], [start_point[2], end_point[2]], [start_point[1], end_point[1]])


# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Z Label')
ax.set_zlabel('Y Label')

#set limits
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)

# Initialize the scatter plot
scatter = ax.scatter([], [], [])

# Initialize the lines
lines = [ax.plot([], [], [], color='blue')[0] for _ in range(len(connections))]

# Initialize the texts
texts = [ax.text(0, 0, 0, i, color='red') for i in range(19)]

# Set up the animation
ani = FuncAnimation(fig, update, frames=None, fargs=(scatter,lines, texts), interval=20)

# Show the plot and start the animation
plt.show()

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
