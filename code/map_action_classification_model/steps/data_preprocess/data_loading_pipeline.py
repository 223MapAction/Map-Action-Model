import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Annotated, Optional, Tuple, List, Dict
from ..data_preprocess.data_transform import get_transform
from zenml import step,pipeline


@step(enable_cache=False)
def create_dataloaders(train_dir:str, valid_dir:str, test_dir:str, batch_size:int) -> Tuple[
    Annotated[DataLoader, "training_dataloader"],
    Annotated[DataLoader, "testing_dataloader"],
    Annotated[int, "num_classes"],
    Annotated[int, "epochs"],
]:
    
    NUM_WORKERS = 2
    
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=get_transform(train=True))
    valid_data = datasets.ImageFolder(valid_dir, transform=get_transform(train=True))
    test_data = datasets.ImageFolder(test_dir, transform=get_transform(train=True))
    
    # Get class names
    class_names = train_data.classes
    num_classes = len(class_names)
    
    # Turn images into data loaders
    training_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    validation_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    testing_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    epochs = 20
    
    
    return training_dataloader, testing_dataloader, num_classes, epochs