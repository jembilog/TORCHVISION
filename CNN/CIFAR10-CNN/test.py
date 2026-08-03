import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = (
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
)

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

test_dataset = datasets.CIFAR10(
    root="/data",
    train=False,
    download=True,
    transform=val_test_transform
)

BATCH_SIZE = 128
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CIFARCNN().to(device)

try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    print("Successfully loaded saved weights from 'best_model.pth'")
except FileNotFoundError:
    print("Error: 'best_model.pth' not found. Ensure the file is in your active execution directory.")
    exit()

model.eval()

all_predictions = []
all_actual = []

print("\nEvaluating entire test dataset split...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_actual.extend(labels.cpu().numpy())
accuracy = accuracy_score(all_actual, all_predictions)
print(f"\nTest Dataset Accuracy: {accuracy*100:.2f}%")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(all_actual, all_predictions))

print("\n--- Detailed Classification Report ---")
print(classification_report(all_actual, all_predictions, target_names=classes))

index = 102
image, label = test_dataset[index]
image_input = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_input)
    probability = torch.softmax(output, dim=1).squeeze()
    prediction = torch.argmax(probability, dim=0).item()

#denormalize image data to format correctly for human visualization
image_np = image.permute(1, 2, 0).numpy()
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
image_np = std * image_np + mean
image_np = np.clip(image_np, 0, 1)

print(f"\n--- Softmax Probabilities for Image Index {index} ---")
for i, p in enumerate(probability):
    print(f"{classes[i]:12s}: {p.item()*100:.2f}%")

plt.imshow(image_np)
plt.title(f"Actual: {classes[label]}\nPredicted: {classes[prediction]}")
plt.axis("off")
plt.show()
