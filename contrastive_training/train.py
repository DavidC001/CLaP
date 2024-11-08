import sys
sys.path.append(".")

from contrastive_training.simsiam.train import train_simsiam
from contrastive_training.simclr.train import train_simclr
from contrastive_training.LASCon.train import train_LASCon
from contrastive_training.MoCo.train import train_moco

train_functions = {
    "simclr": train_simclr,
    "simsiam": train_simsiam,
    "lascon": train_LASCon,
    "moco": train_moco
}

def contrastive_train(params, mode, name, device='cuda', datasets=["panoptic"], models_dir="trained_models", datasets_dir="datasets"):
    """
    Train a contrastive model with the given parameters

    Parameters:
        params (dict): Parameters for the model
        mode (str): Mode of the dataset, can be 'simple'/'complete' or 'multi'
        name (str): Name of the model
        device (str): Device to train on
        datasets (list): List of datasets to use
        models_dir (str): Directory to save the model
        datasets_dir (str): Directory where the datasets are stored
    """
    model = params["model"]
    assert model in train_functions, f"Model {model} not found in {train_functions.keys()}"
    
    print(f"Training {model}")

    train_functions[model](
            **params,
            mode=mode,
            model_dir=models_dir,
            dataset_dir=datasets_dir,
            datasets=datasets,
            name = name, 
            device=device,
    )
    
    print(f"{model} training done")
