import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import train_loader, val_loader
from model import create_model

from config import * 

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using:", device)

model = create_model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam( #adam only
#     filter(
#         lambda p: p.requires_grad,
#         model.parameters()
#     ),
#     lr=LEARNING_RATE
# )
optimizer = torch.optim.AdamW( #adamW
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,
    patience=3
)

#train fucntion
def train_one_epoch():
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs= model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+= loss.item()
        predicted = torch.argmax(outputs,dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    return(running_loss / len(train_loader), accuracy)

#val functioin
def validate():
    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs,dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

best_accuracy = 0
patience = 5
counter = 0
#main training loop
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_acc = validate()
    #scheduler.step(val_acc) #->target - higher acc in validation
    print(
        f"""
        Epoch {epoch+1}/{EPOCHS}
        Train Loss: {train_loss:.4f}
        Train Accuracy: {train_acc:.2f}%
        Validation Accuracy: {val_acc:.2f}%
        """
    )
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        counter= 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model Saved")
    else:
        counter += 1
        print(f"No improvement. Early stopping counter: {counter}/{patience}")
        if counter > patience:
            print(f"\nEarly stopping triggered! Training stopped at epoch {epoch+1}.")
            break
