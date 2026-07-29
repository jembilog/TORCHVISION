import torch 
import torchvision
import torchvision.transforms as transforms


transform = transforms.ToTensor()

#load dataset
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
print("Training Images:", len(train_dataset))
print("Testing Images :", len(test_dataset))

#first image

image , label = train_dataset[0]
# print("\nImage Shape:", image.shape)
# print("Label:", label)
# print("\nPixel Tensor:")
# print(image)
# counter = 0
# for iamge, label in train_dataset:

#     if counter >= 10:
#         break
#     print(
#         f"Image Shape: {image.shape} | "
#         f"Label : {label}"
#     )
#     counter +=1

labels_list = []

for i, (image, label) in enumerate(train_dataset):
    if i >= 10:
        break
    labels_list.append(label)
print("First 10 labels as an array:")
print(labels_list)
