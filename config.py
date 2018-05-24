class Config:
    
    # directories
    summary_dir = "summary/ResNet50"
    checkpoint_dir = "checkpoints/c1"
    
    num_classes = 4
    
    train_on_subset = True
    subset_size = 200
    
    
    model_type = 'ResNet50'
    available_model_types = {'LeNet', 'ResNet50'}
    learning_rate = 1e-4
    train_val_split = 0.8
    batch_size = 8
    num_epochs = 100
    
    max_to_keep = 1
    save_models = False
    