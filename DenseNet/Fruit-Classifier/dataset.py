import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from config import *

#train transform
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

#validation and test uses real images and not the altered ones so they are just the same transform
val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]    
    )
])

#create datasets
train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_test_transform
)
test_dataset=  datasets.ImageFolder(
    root=TEST_DIR,
    transform=val_test_transform
)

#dataloaders
#for training
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader=  DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
