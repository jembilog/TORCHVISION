import torch 
import torch.nn as nn

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
model = DigitClassifier()
images = torch.rand(64,1,28,28)
output = model(images)
print("Input Shape :", images.shape)
print("Output Shape:", output.shape)
