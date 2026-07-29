import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device",device)

transform = transforms.ToTensor()

#load_dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

#data_loader
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

#neural
class DigitClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = DigitClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for image, labels in train_loader:
        images = image.to(device)
        labels = labels.to(device)
        outputs= model(images)
        loss =  criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = torch.argmax(outputs,dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Accuracy: {accuracy*100:.2f}%"
    )

#testing
model.eval()
predictions = []
actual = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        predictions.extend(predicted.cpu().numpy())
        actual.extend(labels.cpu().numpy())
accuracy = accuracy_score(actual, predictions)
print(f"Test Accuracy: {accuracy*100:.2f}%")
torch.save(model.state_dict(), "mnist_model.pth")
print("\nModel Saved Successfully!")
print(labels)
