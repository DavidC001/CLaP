import sys
sys.path.append(".")

from contrastive_training.simsiam.train import train_simsiam
from contrastive_training.simclr.train import train_simclr
from contrastive_training.LASCon.train import train_LASCon

train_functions = {
    "simclr": train_simclr,
    "simsiam": train_simsiam,
    "lascon": train_LASCon,
    "moco": lambda **args : NotImplementedError
}

def contrastive_train(params, name, device='cuda', datasets=["panoptic"], models_dir="trained_models", datasets_dir="datasets"):
    model = params["model"]
    assert model in train_functions, f"Model {model} not found in {train_functions.keys()}"
    
    print(f"Training {model}")

    train_functions[model](
            **params,
            model_dir=models_dir,
            dataset_dir=datasets_dir,
            datasets=datasets,
            name = name, 
            device=device,
    )
    
    print(f"{model} training done")
