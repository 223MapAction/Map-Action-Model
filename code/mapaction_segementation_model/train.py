import torch
from training_pipeline import train
from data_loader import mapaction_test_data_loader, mapction_data_loader
import model



loss_fn = torch.nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

results = train(model=model.mapaction_instance_segmentation_model(num_classes=),train_dataloader=mapction_data_loader,
                test_dataloader=,
                epochs=,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device3)