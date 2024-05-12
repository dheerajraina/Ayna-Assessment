from utils import plot_images
from model2 import hugging_face_stable_diffusion_pipeline
from super_resolution import esrgan_upscaler
from settings import prompt
from utils import plot_images


def pipeline(generator, upscaler, prompt):
    generated_image = generator(prompt)
    sr_image = upscaler(generated_image)
    return sr_image


if __name__ == "__main__":

    image = pipeline(hugging_face_stable_diffusion_pipeline,
                     esrgan_upscaler, prompt)
    plot_images(images=image)
