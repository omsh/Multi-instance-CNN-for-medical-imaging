class Config:

    # directories
    summary_dir = "summary/misi_227_test_2e4_lr25"
    checkpoint_dir = "checkpoints/misi_224_test_2e4_lr25"

    # hardware parameters
    num_parallel_cores = 8
    
    gpu_address = '/device:GPU:0'
    
    # DataLoader parameters
    dataloader_type = 'DatasetFileLoader'
    available_dataloader_types = {'DatasetLoader', 'DatasetFileLoader'}

    # Dataset parameters
    num_classes = 4
    dataset_size = 400
    image_h = 1536
    image_w = 2048
    channels = 3

    # Augmentation

    #  This is pre-augmentation random rotations for images (applied before patching)
    #  angles in degrees and rotation is counter-clock-wise
    
    rotation_angles = [0]
    
    # Rotation for patches: If True, random roations are done on patches
    random_rotation_patches = True
    interpolation = 'NEAREST' 
    available_interpolation = {'NEAREST', 'BILINEAR'} 
    
    # Color Augmentation
    random_contrast = False
    random_hue = False
    random_brightness = False
    random_saturation = False
    

    # training on a subset of the data, for testing and prototyping
    train_on_subset = False
    subset_size = 40

    # patch geneartion
    train_on_patches = True
    patch_size = 227
    
    # to be computed afterwards, needed for  validation and diff patching options
    patch_count = -1
    
    patch_generation_scheme = 'random_crops'
    available_patch_generation_schemes = {'sequential_full', 'sequential_randomly_subset', 'random_crops'}

    
    patches_overlap = 0 # valid only for types 1 and 2, 0.1 --> 10% overlap
    
    n_random_patches = 15  # use this one as a generic n_patches for types 2 and 3
    random_seed = 1
    
    
    # Training parameters
    train_val_split = 0.75
    
    batch_size = 4
    
    num_epochs = 200
    
    # Multiple Instance
    #multiple_instance = True
    mode = 'si_mi_branch'
    available_modes = {'si_branch', 'mi_branch', 'si_mi_branch'}
    beta = 0.5
    
    pooling = 'average'
    available_pooling_functions = {'average', 'max', 'lse'}

    # Optimizer parameters
    optimizer_type = 'Adam'
    available_optimizers = {'Adam', 'GradientDescentOptimizer', 'MomentumOptimizer'}
    
    optim_params = {
        'learning_rate': 2e-4
    }
    
    # learning rate scheduler (decay) type and parameters
    lr_scheduler_type = 'exponential_decay'
    available_lr_schedulers = {'none', 'exponential_decay', 'natural_exp_decay', 'inverse_time_decay'}
    
    lr_scheduler_params = {
        'learning_rate': optim_params['learning_rate'],   # this is the starting learning rate
        'decay_steps': 25 * (dataset_size * train_val_split // batch_size),   # number of steps to wait for before decaying the learning rate
        'decay_rate': 0.5,       # the rate by which the learing rate is decayed
        'staircase': True    # whether to decay discretely or continuously
    }
       
    # Model parameters

    model_type = 'ResNet50'
    available_model_types = {'LeNet', 'ResNet18', 'ResNet50', 'AlexNet', 'Inception', 'ResNeXt'}

    # Model saving parameters
    max_to_keep = 1
    save_models = False
