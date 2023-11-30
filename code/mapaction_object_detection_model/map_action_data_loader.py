import os
import torch
from map_action_tranform import get_transform
from utility import MapActionDataset
from torch.utils.data import DataLoader, Subset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_WORKERS = os.cpu_count()

image_Path = "C:\\Users\\Wakfu\\Downloads\\Compressed\\project-6-at-2023-11-29-00-47-1b67ac6d\\"
annotation_path = "C:\\Users\\Wakfu\\Downloads\\Compressed\\project-6-at-2023-11-29-00-47-1b67ac6d\\result.json"

map_action_dataset = MapActionDataset(image_Path,annotation_path, get_transform(True))
map_action_test_dataset = MapActionDataset(image_Path, annotation_path, get_transform(False))

indices = torch.randperm(len(map_action_dataset)).tolist()
dataset = Subset(map_action_dataset, indices[:7])
test_dataset = Subset(map_action_test_dataset, indices[7:])

def collate_fn(batch):
    return tuple(zip(*batch))


map_action_data_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

map_action_test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


