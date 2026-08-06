import torch
from sklearn.metrics import classification_report,confusion_matrix
from dataset import test_loader, test_dataset
from model import create_model
from config import *
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:",device)

model = create_model()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()
all_predictions = []
all_labels = []
#testing
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs,dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
classes = test_dataset.classes
print(classification_report(all_labels,all_predictions,target_names=classes))
cm = confusion_matrix(all_labels,all_predictions)
print(cm)
plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
