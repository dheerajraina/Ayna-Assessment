import tensorflow as tf
import keras_cv
import keras
import matplotlib.pyplot as plt
from settings import ESRGAN_SAVED_MODEL_PATH
import os
import time
from PIL import Image
import tensorflow_hub as hub
import numpy as np
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

prompt = "Color picture of a beautiful scandinavian girl in high resolution"


def stable_diffusion_image_generator(prompt: str, batch_size=1):
    model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    images = model.text_to_image(prompt, batch_size=batch_size)

    return images


def save_image(image_array: np.array, output_name: str):
    image = Image.fromarray(image_array)
    image.save(f"{output_name}.jpg")


def esrgan_upscaler(org_image):
    model_path = ESRGAN_SAVED_MODEL_PATH
    image = tf.cast(org_image, tf.float32)
    image = tf.expand_dims(image, 0)
    srmodel = hub.load(model_path)
    sr_image = srmodel(image)
    sr_image = tf.squeeze(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)
    sr_image = np.array(sr_image)

    return sr_image


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


images = stable_diffusion_image_generator(prompt)
sr_image = esrgan_upscaler(images[0])
plot_images(sr_image)
