import sys
sys.path.append(".")

import os
import json
from pose_estimation.models import models, getPoseEstimModel
from pose_estimation.functions import get_optimizer
from dataloaders.datasets import out_joints 
from pose_estimation.train import train
from pose_estimation.utils import getLatestModel, getDatasetLoader

def parseArgs(args):
    """
    Parse the arguments for the pose estimation model.

    Parameters:
    - args: dict, arguments for the pose estimation model

    Returns:
    - args: dict, updated arguments for the pose estimation model
    """
    assert 'architecture' in args, 'estimator head architecture not found in args'
    assert 'name' in args, 'name not found for args'
    assert 'pretrained_name' in args, 'pretrained model name not found in args'

    default_args = {
        'train': True,
        'batch_size': 1024,
        'learning_rate': 0.01,
        'weight_decay': 0.000001,
        'momentum': 0.9,
        'epochs': 20,
        'save_every': 10
    }

    args = {**default_args, **args}
    
    return args

def pose_estimation( args, device='cpu', models_dir="trained_models", datasets_dir="datasets", base_model="resnet18"):
    """
    Train the pose estimation model.

    Parameters:
    - args: dict, arguments for the pose estimation model
    - device: str, device, default is 'cpu'
    - models_dir: str, directory to save the trained models, default is 'trained_models'
    - datasets_dir: str, directory to save the datasets, default is 'datasets'
    - base_model: str, base model, default is 'resnet18'

    """
    #skip if specified
    if "skip" in args and args["skip"]:
        print("Skipping pose estimation training")
        return
    
    assert "dataset" in args, "dataset not found in args"

    train_loader, val_loader, test_loader = getDatasetLoader(dataset=args["dataset"], batch_size=args["batch_size"], datasets_dir=datasets_dir, base_model=base_model)
    
    for model in models:
        if model in args:
            params = parseArgs(args[model])
            params["dataset"] = args["dataset"]
            params["batch_size"] = args["batch_size"]

            if (params["train"]):
                print(f"Training {model}")

                try:
                    #save parameters to file
                    with open(f"{models_dir}/{params['name']}_{model}_params.json", 'w') as f:
                        json.dump(params, f)

                    pretrained = getPoseEstimModel(
                            path = getLatestModel(os.path.join(models_dir, params['pretrained_name'])),
                            model_type=model,
                            layers=params['architecture'],
                            out_dim=out_joints[args['dataset']]*3,
                            device=device,
                            base_model=base_model
                        )
                    # print(pretrained)
                        
                    optim, scheduler = get_optimizer(
                            net=pretrained,
                            learning_rate=params["learning_rate"],
                            momentum=params["momentum"],
                            weight_decay=params["weight_decay"],
                            T_max=params["epochs"]
                        )

                    train(
                            model=pretrained,
                            optimizer= optim,
                            scheduler=scheduler,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=test_loader,
                            epochs=params["epochs"],
                            save_every=params["save_every"],
                            device=device,
                            model_dir=models_dir,
                            name=params["name"]
                        )
                    
                    print(f"{model} training done")

                except Exception as e:
                    print(f"Error training {model}: {e}")
                    continue
                    

if __name__ == '__main__':
    args = {
        'dataset': 'skiPose',
        'skip': False,
        'batch_size': 1024,
        'simclr': {
            'architecture': [512, 256],
            'name': 'simclr_estim',
            'pretrained_name': 'simclr',
            'train': True,
            'learning_rate': 0.01,
            'weight_decay': 0.000001,
            'momentum': 0.9,
            'epochs': 20,
            'save_every': 10
        }
    }

    pose_estimation(args, device="cuda")
