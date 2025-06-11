import struct
import numpy as np
import matplotlib.pyplot as plt
from utils.transforms import load_emnist_mapping

def load_ubyte_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"Magic: {magic}, Number of Images: {num}, Size: {rows}x{cols}")
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_ubyte_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        print(f"Magic: {magic}, Number of Labels: {num}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Example usage:
images_path = './data/EMNIST/raw/emnist-byclass-train-images-idx3-ubyte'
labels_path = './data/EMNIST/raw/emnist-byclass-train-labels-idx1-ubyte'

images = load_ubyte_images(images_path)
labels = load_ubyte_labels(labels_path)

mapping = load_emnist_mapping()

# Show the first 10 images
for i in range(20):
    image = images[i]
    label = int(labels[i])

    # image = np.transpose(image, (1, 0)) # Rotate 90 degrees counter-clockwise
    # image = np.flip(image, axis=1)

    char = mapping[label]

    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {labels}, character: {char}')
    plt.axis('off')
    plt.show()
