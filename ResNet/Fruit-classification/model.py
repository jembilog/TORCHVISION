import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def create_model():
    #load pretrained resnet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    #freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(
        512,
        NUM_CLASSES
    )

    return model
