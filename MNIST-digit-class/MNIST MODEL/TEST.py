import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
model = DigitClassifier().to(device)
model.load_state_dict(torch.load("mnist_model.pth"))


# model.eval()
# #chooosing image
# image, label = test_dataset[0]
# print("Actual Label:",label)
# #adding batch dimesion
# image = image.unsqueeze(0)
# image = image.to(device)
#prediction
# with torch.no_grad():
#     output = model(image)
#     probabilities = torch.softmax(output, dim=1)
#     prediction = torch.argmax(output,dim=1)
# print("\nPrediction Probabilities")
# print("\nPrediction Probabilities")

# for i in range(10):
#     print(f"{i}: {probabilities[0][i]:.4f}")

# print("\nPredicted Digit:", prediction.item())


#predict all test images
# predictions = []
# actual_labels = []

# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(device)
#         outputs = model(images)
#         predicted =  torch.argmax(outputs, dim=1)
#         predictions.extend(predicted.cpu().numpy())
#         actual_labels.extend(labels.numpy())
# print("\n========================================")
# print("First 100 Actual Labels")
# print(actual_labels[:100])

# print("\nFirst 100 Predicted Labels")
# print(predictions[:100])
# print("========================================")



#predict one image
index = 190
image,actual = test_dataset[index]
display_image = image.squeeze()
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probabilities=  torch.softmax(output,dim=1)
    prediction = torch.argmax(output,dim=1).item()
#print predictions
print(f"Image Index      : {index}")
print(f"Actual Label     : {actual}")
print(f"Predicted Label  : {prediction}")
if actual == prediction:
    print("Result           : Correct")
else:
    print("Result           : Wrong")
print("\nPrediction Probabilities\n")

for i in range(10):
    print(f"Digit {i}: {probabilities[0][i].item():.6f}")


plt.figure(figsize=(4,4))

plt.imshow(display_image, cmap="gray")

plt.title(
    f"Actual: {actual} | Predicted: {prediction}"
)

plt.axis("off")

plt.show()
