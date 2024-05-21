# Human Pose Estimation using Contrastive Learning
A Course Project as part of course final exam Submitted By:
- Davide Cavicchini
- Sofia Lorengo
- Alessia Pivotto

To execute the project run contrastive_HPE.py script with argument '--experiment file' with file being a json like "experiments/cluster.json" 

## Abstract

## Dataset

## Methodologies

### SimCLR

### SimSiam

### MoCo

### LASCon

## Project Structure
```
  ├── contrastive_training   # contrastive pre-training files
  │   ├── MoCo               # files for MoCo model and training
  │   │   └── ...
  │   ├── simclr             # files for SimCLR model and training
  │   │   └── ...
  │   └── simsiam            # files for SimSiam model and training
  │   │   └── ...
  │   └── LASCon             # files for LASCon training
  │   │   └── ...
  │   └── visualize_data.py  # script to navigate the generated embedding space
  │   └── train.py           # script used as an interface to start the training for the different models
  │   └── ...                # other files
  ├── dataloaders            # scripts used to load the datasets
  │   └── ...
  ├── experiments            # folder containing the JSON used as configuration for different training runs
  │   └── ...
  ├── pose_estimation        # pose estimation models training scripts
  │   ├── train.py           # script used as an interface to start the training on the different pre-trained models
  │   └── ...
  ├── contrastive_HPE.py     # main file
  ├── ...           
  └── README.md          # this file
```
## Results
