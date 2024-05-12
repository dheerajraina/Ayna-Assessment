from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def save_image(image_array: np.array, output_name: str):
    image = Image.fromarray(image_array)
    image.save(f"{output_name}.jpg")


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
