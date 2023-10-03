from PIL import Image

from stable_diffusion.stable_diffusion import StableDiffusion

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
img = model.text_to_image(
    "a cute girl.",
    num_steps=25,
    seed=123456,
)
Image.fromarray(img[0]).save("girl.jpg")
print("Saved at girl.jpg")
