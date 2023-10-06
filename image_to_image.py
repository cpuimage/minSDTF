from PIL import Image

from stable_diffusion.stable_diffusion import StableDiffusion

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
img = model.image_to_image(
    "a cute girl.",
    unconditional_guidance_scale=7.5,
    reference_image="/path/to/a_girl.jpg",
    reference_image_strength=0.8,
    num_steps=50,
)
Image.fromarray(img[0]).save("out.jpg")
print("Saved at out.jpg")
