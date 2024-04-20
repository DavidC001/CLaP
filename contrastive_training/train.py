import sys
sys.path.append(".")

from contrastive_training.simsiam.train import train_simsiam
from contrastive_training.simclr.train import train_simclr

train_functions = {
    "simclr": train_simclr,
    "simsiam": train_simsiam
}

def contrastive_train(model, params, device='cuda', datasets=["panoptic"], models_dir="trained_models", datasets_dir="datasets"):
    assert model in train_functions, f"Model {model} not found in {train_functions.keys()}"
    
    print(f"Training {model}")

    train_functions[model](
            model_dir=models_dir,
            dataset_dir=datasets_dir,
            datasets=datasets,
            name = params["name"], 
            batch_size=params["batch_size"],
            device=device,
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            momentum=params["momentum"],
            t=params["temperature"],
            epochs=params["epochs"],
            save_every = params["save_every"]
    )
    
    print(f"{model} training done")


if __name__ == "__main__":
    contrastive_train()