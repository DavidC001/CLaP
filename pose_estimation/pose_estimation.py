from torch.utils.tensorboard import SummaryWriter

def pose_estimation( args, device='cpu', models_dir="trained_models", datasets_dir="datasets"):
    _, _, _, train_loader, val_loader, test_loader = get_data(batch_size)

    # pose estimation TODO
    #simclr_path = 'trained_models/ver1.pt'
    simclr_path = 'trained_models/simclr/simclr_epoch_25.pth'

    simclr = get_simclr_net()

    main(simclr_path, simclr, name="sim_2layer_2.1", epochs=40, learning_rate=0.02, device=device)
