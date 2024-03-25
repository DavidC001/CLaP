import sys
sys.path.append(".")

from contrastive_training.simsiam.train import train_simsiam
from contrastive_training.simclr.train import train_simclr


def contrastive_train(device='cuda:0', dataset="panoptic",
                      simsiam=True, name_siam = "simsiam",
                      batch_size_siam=1024, lr_siam=0.01, wd_siam=0.000001, momentum_siam=0.9, t_siam=0.6, epochs_siam=100,
                      simclr=True, name_clr = "simclr",
                      batch_size_clr=1024, lr_clr=0.01, wd_clr=0.000001, momentum_clr=0.9, t_clr=0.6, epochs_clr=100
                      ):
    if simclr:
        print("Training SimCLR")
        train_simclr(
            name = name_clr, 
            batch_size=batch_size_clr, 
            device=device, 
            learning_rate=lr_clr, 
            weight_decay=wd_clr, 
            momentum=momentum_clr, 
            t=t_clr, 
            epochs=epochs_clr,
            dataset=dataset
        )
        print("SimCLR training done")
    
    if simsiam:
        print("Training SimSiam")
        train_simsiam(
            name = name_siam, 
            batch_size=batch_size_siam, 
            device=device, 
            learning_rate=lr_siam, 
            weight_decay=wd_siam, 
            momentum=momentum_siam, 
            t=t_siam, 
            epochs=epochs_siam,
            dataset=dataset
        )
        print("SimSiam training done")


if __name__ == "__main__":
    contrastive_train()