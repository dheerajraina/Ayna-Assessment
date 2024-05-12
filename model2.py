import torch
from diffusers import StableDiffusionPipeline
from settings import prompt, ESRGAN_SAVED_MODEL_PATH
import numpy as np


def hugging_face_stable_diffusion_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]

    return np.array(image)
