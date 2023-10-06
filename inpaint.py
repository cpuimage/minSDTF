from PIL import Image

from stable_diffusion.stable_diffusion import StableDiffusion

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
img = model.inpaint(
    "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.",
    reference_image="/path/to/dog.jpg",
    inpaint_mask="/path/to/dog_mask.png",
    mask_blur_strength=5,
    unconditional_guidance_scale=8.0,
    reference_image_strength=0.9,
    num_steps=50,
)
Image.fromarray(img[0]).save("out.jpg")
print("Saved at out.jpg")
