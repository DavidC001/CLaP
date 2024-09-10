import os
import json
import sys
sys.path.append(".")

from contrastive_training.train import contrastive_train
from contrastive_training.utils import load_datasets

models = ["simsiam", "simclr", "MoCo", "LASCon"]

def check_arguments_contrastive(args):
    #name is required
    assert 'name' in args, 'name not found for args'

    default_args = {
        "train": True,
        "batch_size": 256,
        "learning_rate": 0.02,
        "weight_decay": 0.000001,
        "momentum": 0.9,
        "temperature": 0.6,
        "epochs": 5,
        "save_every": 10
    }

    args = {**default_args, **args}
    
    return args

def contrastive_pretraining(args, device='cuda', models_dir="trained_models", datasets_dir="datasets", base_model='resnet18'):
    default_args = {
        "skip": False,
        "datasets": ["panoptic"],
        "use_complete_pairs": True,
        "drop_pairs": [0]
    }

    if 'datasets' in args and 'drop_pairs' in args:
        assert len(args['datasets']) == len(args['drop_pairs']), "Number of datasets and drop pairs should be the same"
    args = {**default_args, **args}
    if len(args['datasets']) != len(args['drop_pairs']):
        args['drop_pairs'] = [0] * len(args['datasets'])
    
    if args['skip']:
        print("Skipping contrastive training")
        return

    load_datasets(args['datasets'], dataset_dir=datasets_dir, base_model=base_model, 
                    use_complete=args['use_complete_pairs'], drop=args['drop_pairs'])

    print("Contrastive training")

    for model in models:
        if model in args:
            params = args[model]
            params["datasets"] = args['datasets']
            params = check_arguments_contrastive(params)
            
            if params['train']:
                try:
                    #save parameters to file
                    with open(f"{models_dir}/{params['name']}_{model}_params.json", 'w') as f:
                        json.dump(params, f)
                    contrastive_train(device=device,  model=model, params=params, datasets=args['datasets'], models_dir=models_dir, datasets_dir=datasets_dir, base_model=base_model)
                except Exception as e:
                    print(f"Error in training {model}: {e}") 
            