import tensorflow as tf
import keras_cv
import keras
import matplotlib.pyplot as plt
from utils import plot_images


def stable_diffusion_image_generator(prompt: str, batch_size=1):
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    images = model.text_to_image(prompt, batch_size=batch_size)

    return images
