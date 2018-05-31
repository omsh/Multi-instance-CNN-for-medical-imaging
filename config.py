class Config:
    
    # directories
    summary_dir = "summary/testing"
    checkpoint_dir = "checkpoints/c1"
    
    # hardware parameters
    num_parallel_cores = 8
    
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
    patch_size = 224
    
    # patch sampling (not used yet)
    n_patches_to_sample_per_image = 32
    
    # Training parameters
    train_val_split = 0.8
    
    # Optimizer parameters
    
    # Model parameters
    model_type = 'ResNet50'
    available_model_types = {'LeNet', 'ResNet50', 'AlexNet'}
    
    learning_rate = 1e-4
    batch_size = 128
    num_epochs = 2
    
    # Model saving parameters
    max_to_keep = 1
    save_models = False
    
    