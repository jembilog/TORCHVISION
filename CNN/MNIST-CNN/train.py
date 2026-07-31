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

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #convolutional and pooling layers
        self.convolution1 = nn.Conv2d(
            1,
            32,
            kernel_size=3, 
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.convolution2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.flatten = nn.Flatten()
        #basic neural
        self.fc1 = nn.Linear(64*7*7,128)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.convolution2(x)
        x = self.relu2(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)

        return x
model =  CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    print(
        f"Epoch {epoch+1} | "
        f"Loss: {train_loss:.4f} | "
        f"Accuracy: {train_accuracy*100:.2f}%"
    )
# plt.imshow(
#     images[0].squeeze(),
#     cmap="gray"
# )
# plt.title(

#     f"Label : {labels[0].item()}"

# )

# plt.show()
