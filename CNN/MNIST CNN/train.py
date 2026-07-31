import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt

device = torch.device(
    "cuda" if  torch.cuda.is_available() else "cpu"
)
print(device)

#preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,),
    )
])

#datasets
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

#loader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(train_loader)) #if u want to display just change this in to test_loader or simply chnage the train variable in train_dataset into false so that the output will be the same on its index 
print(images.shape)
print(labels.shape)

# for looking pcitures
# plt.imshow(
#     images[0].squeeze(), 
#     cmap="gray"
# )
# plt.title(
#     f"Label : {labels[0].item()}"
# )
# plt.show()
