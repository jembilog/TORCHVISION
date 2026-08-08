import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def create_model():
    #load pretrained MobileNetV2
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    #freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    #unfreeze the last feature block #-> uncomment if u need fine tunin'
    # for param in model.features[-1].parameters():
    #     param.requires_grad = True

    #Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    #make classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model
