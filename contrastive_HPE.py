import sys
sys.path.append('.')

from contrastive_training.train import contrastive_train

print("Test")
contrastive_train(simclr=False, simsiam=False)
