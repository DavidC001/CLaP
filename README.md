# CLaP: Contrast Label Predict for Human Pose Estimation
![image](https://github.com/user-attachments/assets/d8b07e5d-b4c4-4be2-be88-f70f9fcfbe49)
This repository contains the implementation of CLaP, a framework that leverages contrastive learning for human pose estimation in data-constrained domains.

## Overview
CLaP combines contrastive pre-training with supervised pose estimation to improve performance when limited training data is available. The framework consists of:

1. Contrastive Pre-training: Multiple state-of-the-art contrastive learning approaches including:
  - SimCLR
  - SimSiam
  - MoCo
2. Pose Estimation: Supervised training for pose estimation using the pre-trained representations

## Project Structure
```
├── contrastive_training/    # Contrastive pre-training implementations
│   ├── MoCo/               # Momentum Contrast implementation
│   ├── simclr/             # SimCLR implementation
│   ├── simsiam/            # SimSiam implementation 
│   ├── LASCon/             # LASCon implementation (not included in the paper)
│   ├── visualize_data.py   # Tools for visualizing embeddings
│   └── train.py           
├── dataloaders/            # Dataset loading utilities
├── experiments/            # Training configuration files
├── pose_estimation/        # Pose estimation model & training
├── contrastive_HPE.py      # Main training script
└── README.md
```

## Installation
```bash
# Clone repository
git clone https://github.com/username/CLaP.git
cd CLaP

# Install dependencies
pip install -r requirements.txt
```

## Usage
Train a model using the provided  configuration files:
```bash
python contrastive_HPE.py --experiment experiments/resnet18.json
```

Configuration files in experiments contain hyperparameters and model settings:
- `resnet18.json`: ResNet18 backbone configurations
- `resnet50.json`: ResNet50 backbone configurations
- `moco.json`: MoCo-specific configurations

## Datasets
The SkiPose dataset is not publicly available, but can be requested from the authors.

## Key Features
Multiple contrastive learning approaches (SimCLR, SimSiam, MoCo, LASCon)
Flexible backbone architectures (ResNet18, ResNet50)
Multi-view and data augmentation strategies
TensorBoard integration for training visualization
Configurable training parameters via JSON files

# Citation
If you find this work useful, consider citing the paper
```
@InProceedings{Cavicchini_2025_WACV,
    author    = {Cavicchini, Davide and Pivotto, Alessia and Lorengo, Sofia and Rosani, Andrea and Garau, Nicola},
    title     = {CLaP - Contrast Label Predict: a quest for cheaper labeling in 3D human pose estimation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {February},
    year      = {2025},
    pages     = {1276-1284}
}
```

# Authors
- [Davide Cavicchini](https://github.com/DavidC001)
- [Alessia Pivotto](https://github.com/AlessiaPivotto)
- [Sofia Lorengo](https://github.com/sofy01)
- Andrea Rosani
- Nicola Garau
