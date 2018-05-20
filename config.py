class Config:
    
    # directories
    summary_dir = "summary/s1"
    checkpoint_dir = "checkpoints/c1"
    
    num_classes = 4
    
    train_on_subset = True
    subset_size = 240
    
    learning_rate = 1e-4
    train_val_split = 0.7
    batch_size = 8
    num_epochs = 20
    
    max_to_keep = 1
    