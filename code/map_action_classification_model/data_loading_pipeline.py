import torch
from torchvision import datasets
from torch.utils.data import DataLoader

def create_dataloaders(train_dir, valid_dir, test_dir, transform, batch_size, num_workers):
    
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    
    # Turn images into data loaders
    training_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    validation_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    testing_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return training_dataloader, testing_dataloader, class_names