class Config:

    # directories
    summary_dir = "summary/ResNet50_fastloader_with_patching_magda"
    checkpoint_dir = "checkpoints/ResNet50_fastloader_with_patching_magda"

    # hardware parameters
    num_parallel_cores = 8
    
    gpu_address = '/device:GPU:0'
    
    # DataLoader parameters
    dataloader_type = 'DatasetFileLoader'
    available_dataloader_types = {'DatasetLoader', 'DatasetFileLoader'}

    # Dataset parameters
    num_classes = 4

    # Augmentation

    #  This is pre-augmentation random rotations for images (applied before patching)
    # angles in degrees and rotation is counter-clock-wise
    rotation_angles = [0, 90, 180, 270]

    # training on a subset of the data, for testing and prototyping
    train_on_subset = True
    subset_size = 40

    # patch geneartion (currently on the fly, can be extended to save on the hard disk)
    train_on_patches = True
    patch_size = 672
    
    pick_random_patches = False
    pick_n_random_patches = 12
    random_seed = 1

    # Training parameters
    train_val_split = 0.75

    # Optimizer parameters
    optimizer_type = 'Adam'
    available_optimizers = {'Adam', 'GradientDescentOptimizer', 'MomentumOptimizer'}
    
    optim_params = {
        'learning_rate': 1e-3
    }
       
    # Model parameters
    model_type = 'ResNet50'
    available_model_types = {'LeNet', 'ResNet50', 'AlexNet', 'Inception'}

    
    batch_size = 5
    num_epochs = 20

    # Model saving parameters
    max_to_keep = 1
    save_models = False
