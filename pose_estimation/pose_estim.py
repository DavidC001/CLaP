import sys
sys.path.append(".")

import os
from pose_estimation.models import models, getPoseEstimModel
from pose_estimation.functions import get_optimizer
from dataloaders.datasets import out_joints 
from pose_estimation.train import train
from pose_estimation.utils import getLatestModel, getDatasetLoader

def parseArgs(args):
    assert 'architecture' in args, 'estimator head architecture not found in args'
    assert 'name' in args, 'name not found for args'
    assert 'pretrained_name' in args, 'pretrained model name not found in args'

    if 'train' not in args:
        args['train'] = True
    if 'batch_size' not in args:
        args['batch_size'] = 1024
    if 'learning_rate' not in args:
        args['learning_rate'] = 0.01
    if 'weight_decay' not in args:
        args['weight_decay'] = 0.000001
    if 'momentum' not in args:
        args['momentum'] = 0.9
    if 'epochs' not in args:
        args['epochs'] = 20
    if 'save_every' not in args:
        args['save_every'] = 10
    
    return args

def pose_estimation( args, device='cpu', models_dir="trained_models", datasets_dir="datasets"):
    #skip if specified
    if "skip" not in args:
        args["skip"] = False
    if args['skip']:
        print("Skipping pose estimation training")
        return
    
    assert "dataset" in args, "dataset not found in args"

    train_loader, val_loader, test_loader = getDatasetLoader(dataset=args["dataset"], batch_size=args["batch_size"], datasets_dir=datasets_dir)
    
    for model in models:
        if model in args:
            params = parseArgs(args[model])
            if (params["train"]):
                print(f"Training {model}")

                try:

                    pretrained = getPoseEstimModel(
                            path = getLatestModel(os.path.join(models_dir, params['pretrained_name'])),
                            model_type=model,
                            layers=params['architecture'],
                            out_dim=out_joints[args['dataset']]*3,
                            device=device
                        )
                    print(pretrained)
                        
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
