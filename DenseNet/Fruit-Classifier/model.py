import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def create_model():
    # Load pretrained DenseNet121
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    #freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    #Replace classifier
    model.classifier = nn.Linear(
        model.classifier.in_features,
        NUM_CLASSES
    )

    #make classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model
