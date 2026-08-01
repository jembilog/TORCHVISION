import numpy as np
from torch import le

image = np.array([
    [1,-1,-1,-1,1],
    [-1,1,-1,1,-1],
    [-1,-1,1,-1,-1],
    [-1,1,-1,1,-1],
    [1,-1,-1,-1,1]
])

kernel = np.array([
    [1,-1,-1],
    [-1,1,-1],
    [-1,-1,1]
])

def conv(image,kernel):
    image_height, image_width  = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    feature_map = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            region = image[i:i+kernel_height, j:j+kernel_width]
            feature_map[i,j] = np.sum(region * kernel)
    return feature_map

result = conv(image, kernel)
print(result)
