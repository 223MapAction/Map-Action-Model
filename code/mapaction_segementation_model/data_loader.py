import torch
import os
from utils import MapActionDataset
from data_transform import get_transform
from torch.uitls.data import DataLoader, Subset

device = torch.device('cuda') if torch.cuda.is_available() else
torch.device('cpu')

num_classes = 
NUM_WORKERS = os.cpu_count()


mapaction_train_dataset = MapActionDataset('', get_transform(train=True))
mapaction_test_dataset = MapActionDataset('', get_transform(train=false))


indices = torch.randperm(len(dataset)).tolist()
dataset = Subset(mapaction_train_dataset, indices[:])
test_dataset = Subset(mapaction_test_dataset, indices[:])




mapction_data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_worker=NUM_WORKERS,
    collate_fn=utils.collate_fn
)

mapaction_test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_worker=NUM_WORKERS,
    collate_fn=utils.collate_fn
)