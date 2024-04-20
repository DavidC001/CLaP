import sys
sys.path.append(".")

from contrastive_training.train import contrastive_train
from contrastive_training.utils import load_datasets

models = ["simsiam", "simclr", "MoCo", "LASCon"]

def check_arguments_contrastive(args):
    #name is required
    assert 'name' in args, 'name not found for args'

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
    if 'temperature' not in args:
        args['temperature'] = 0.6
    if 'epochs' not in args:
        args['epochs'] = 100
    if 'save_every' not in args:
        args['save_every'] = 10
    
    return args

def contrastive_pretraining(args, device='cuda', models_dir="trained_models", datasets_dir="datasets"):
    #skip if specified
    if args['skip']:
        print("Skipping contrastive training")
        return
    
    if 'datasets' not in args:
        args['datasets'] = ["panoptic"]

    load_datasets(args['datasets'], datasets_dir=datasets_dir)

    print("Contrastive training")

    for model in models:
        if model in args:
            params = args[model]
            params = check_arguments_contrastive(params)
            
            if params['train']:
                contrastive_train(device=device,  model=model, params=params, datasets=args['datasets'], models_dir=models_dir, datasets_dir=datasets_dir)
            