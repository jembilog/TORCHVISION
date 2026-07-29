import torch 

image = torch.rand(1,28,28)
print("Single Image Shape:", image.shape)

#batch of images 
images = torch.rand(64,1,28,28)
print("Batch Shape:", images.shape)

print("Batch Size:", images.shape[0])
print("Channels:", images.shape[1])
print("Height:", images.shape[2])
print("Width:", images.shape[3])
