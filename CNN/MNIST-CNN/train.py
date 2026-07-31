import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torch.utils.data import random_split

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

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset,[train_size, val_size])

#loader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# print(images.shape)
# print(labels.shape)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #convolutional and pooling layers
        self.convolution1 = nn.Conv2d(
            1, #input
            32, #no. of filters / will produced feature maps
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
            32, #input
            64, #no. of filters / will produced feature maps
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        #basic neural
        self.fc1 = nn.Linear(64*7*7,128)
        self.relu3 = nn.ReLU()
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
        x = self.relu3(x)
        x = self.fc2(x)

        return x


model =  CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 10

#for early stopping
best_loss = float("inf")
counter = 0
patience = 5

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
    # print(
    #     f"Epoch {epoch+1} | "
    #     f"Loss: {train_loss:.4f} | "
    #     f"Accuracy: {train_accuracy*100:.2f}%"
    # )
    model.eval()

    validation_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            predicted = torch.argmax(outputs,dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    validation_loss /= len(val_loader)
    validation_accuracy = correct / total
    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy*100:.2f}% | "
        f"Val loss: {validation_loss:.4f} | " 
        f"Val Acc: {validation_accuracy*100:.2f}%"
    )

    if validation_loss < best_loss:
        best_loss = validation_loss
        counnter = 0 
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model updated")
    else:
        counnter += 1
        if counnter >= patience:
            print("Early stopping triggered")
            break

#load best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
predictions = []
actual = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs,dim=1)
        predictions.extend(predicted.cpu().numpy())
        actual.extend(labels.cpu().numpy())
accuracy = accuracy_score(actual,predictions)
print("Test Accuracy:",accuracy)
print(confusion_matrix(actual,predictions))
print(classification_report(actual,predictions))

#iterate through test loader array
# images, labels = next(iter(test_loader))
# outputs = model(images.to(device))
# predicted = torch.argmax(outputs,dim=1)
# plt.imshow(images[0].squeeze(),cmap="gray")
# plt.title(f"Actual: {labels[0]} | Predicted: {predicted[0].item()}")
# plt.axis("off")
# plt.show()


#choose image index
index = 1
image, actual_label = test_dataset[index]
image_input = image.unsqueeze(0).to(device)

#predict bitch
model.eval()
with torch.no_grad():
    output = model(image_input)
    probabilities= torch.softmax(output,dim=1).squeeze()
    predicted_label = torch.argmax(output,dim=1).item()
print(f"Dataset Index   : {index}")
print(f"Actual Label    : {actual_label}")
print(f"Predicted Label : {predicted_label}")
print("\nPrediction Probabilities\n")
if actual_label == predicted_label:
    print("Result           : Correct")
else:
    print("Result           : Wrong")

#showing image
#undo normalization
display_image = image.squeeze().numpy()
display_image = (display_image * 0.3081) + 0.1307
plt.figure(figsize=(4,4))
plt.imshow(display_image, cmap="gray")
plt.title(
    f"Actual: {actual_label} | Predicted: {predicted_label}"
)
plt.axis("off")
plt.show()
