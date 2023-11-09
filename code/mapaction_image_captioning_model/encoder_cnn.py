import torch
import torch.nn as nn
import torchvision.moels as models


class EncoderCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EncoderCNN, self).__init__()
        resnet = # yet to be trained
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features