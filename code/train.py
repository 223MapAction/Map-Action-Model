import torch

from model import mapaction_instance_segmentation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes =

model = mapaction_instance_segmentation_model(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.require_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.0002,
    momentum=0.9,
    weights_decay=0.0005
)



