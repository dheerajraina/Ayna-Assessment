import tensorflow as tf
import tensorflow_hub as hub
from settings import ESRGAN_SAVED_MODEL_PATH
import os
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


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
