import torch
import matplotlib.pyplot as plt
from dataset import test_loader, test_dataset
from model import create_model
from config import *

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("Using:", device)
model = create_model()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)

model.eval()
images, labels = next(iter(test_loader))
index = 12
image = images[index]
label = labels[index]
input_image = image.unsqueeze(0).to(device)
with torch.no_grad():

    output = model(input_image)
    probabilities = torch.softmax(output,dim=1)
    confidence, prediction = torch.max(probabilities,1)
classes = test_dataset.classes
print("Actual:",classes[label])
print("Predicted:",classes[prediction.item()])
print("Confidence:",confidence.item()*100,"%")
print("\nProbabilities:")

for i, prob in enumerate(probabilities[0]):
    print(f"{classes[i]}: {prob.item()*100:.2f}%")

#displaying the image
mean = torch.tensor(
    [0.485,0.456,0.406]
).view(3,1,1)
std = torch.tensor(
    [0.229,0.224,0.225]
).view(3,1,1)
image = image * std + mean
image = image.permute(1,2,0)
plt.imshow(image)

plt.title(
    f"""
Actual: {classes[label]}

Predicted:
{classes[prediction.item()]}
"""
)

plt.axis("off")

plt.show()
