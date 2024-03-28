import sys
sys.path.append('.')

import argparse
import json
import torch

from contrastive_training.train import contrastive_pretraining

import random
random.seed(0)

def main(args):
    #read experiment json file
    with open(args.experiment) as f:
        data = json.load(f)
    
    #get the required parameters
    device = data['device']
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = data['dataset']
    if dataset == None:
        dataset = 'panoptic'
    models_dir = data['models_dir']
    if models_dir == None:
        models_dir = 'trained_models'

    contrastive_pretraining(args=data["contrastive"], device=device, dataset=dataset, models_dir=models_dir)

    


if __name__ == "__main__":
    #get experment file name from command line (required)
    parser = argparse.ArgumentParser(description='Contrastive training')
    parser.add_argument('--experiment', type=str, help='Path to the experiment json file', required=True)
    args = parser.parse_args()
    
    main(args)