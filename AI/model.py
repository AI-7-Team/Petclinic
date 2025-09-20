import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes, pretrained=True):
    model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model