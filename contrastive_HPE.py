import sys
sys.path.append('.')

import argparse
import json
import torch
import numpy as np
import os

#disable the warning
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

from contrastive_training.contrastive import contrastive_pretraining
from pose_estimation.pose_estim import pose_estimation

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def main(args):
    #read experiment json file
    with open(args.experiment) as f:
        data = json.load(f)
    
    default_args = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_dir": "trained_models",
        "datasets_dir": "datasets",
        "base_model": "resnet18"
    }
    args = {**default_args, **data}

    #get the required parameters
    device = data['device']
    models_dir = data['models_dir']
    datasets_dir = data['datasets_dir']

    # if it doesn't exist, create the directory to save the models
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        os.makedirs(models_dir + "/resnet50")
    
    contrastive_pretraining(args=data["contrastive"], device=device, models_dir=models_dir, datasets_dir=datasets_dir)
    pose_estimation(args=data["pose_estimation"], device=device, models_dir=models_dir, datasets_dir=datasets_dir)



if __name__ == "__main__":
    #get experment file name from command line (required)
    parser = argparse.ArgumentParser(description='Contrastive training')
    parser.add_argument('--experiment', type=str, help='Path to the experiment json file', required=True)
    args = parser.parse_args()

    main(args)