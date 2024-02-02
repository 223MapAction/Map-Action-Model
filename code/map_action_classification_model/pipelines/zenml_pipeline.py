import os
import torch
import mlflow
from zenml import pipeline, step
from steps import download_and_organize_data
from steps import get_transform
from steps import m_a_model
from steps import train_model
from steps import test_step
from steps import create_dataloaders



@pipeline
def zenml_training_pipeline():
    
    train_dir, valid_dir, test_dir, batch_size = download_and_organize_data()
    training_dataloader, testing_dataloader, num_classes, epochs = create_dataloaders(train_dir = train_dir,
                                                                              valid_dir = valid_dir, 
                                                                              test_dir = test_dir,
                                                                              batch_size = batch_size)
    model= m_a_model(num_classes)
    model, results = train_model(model ,training_dataloader, epochs)
    test_loss, test_acc = test_step(model, testing_dataloader)
    
    
    