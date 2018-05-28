class Config:
    
    # directories
    summary_dir = "summary/alexnet"
    checkpoint_dir = "checkpoints/alexnet"
    
    num_classes = 4
    
    train_on_subset = True
    subset_size = 40
    
    
    model_type = 'AlexNet'
    available_model_types = {'LeNet', 'ResNet50', 'AlexNet'}
    learning_rate = 1e-4
    train_val_split = 0.8
    batch_size = 8
    num_epochs = 20
    
    max_to_keep = 1
    save_models = False
