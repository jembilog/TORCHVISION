import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def create_model():
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    #freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

        # Unfreeze last feature block
    for param in model.features[-1].parameters():
        param.requires_grad = True

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    for param in model.classifier.parameters():
        param.requires_grad = True
    return model
