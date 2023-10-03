# minSDTF

Stable Diffusion V1.5 Inference With PyTorch Weights in TensorFlow 2

#### Using pip without a virtual environment

Install dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Using the Python interface

If you installed the package, you can use it as follows:

```python 
import cv2
import numpy as np
from PIL import Image
from stable_diffusion.stable_diffusion import StableDiffusion

# for load civitai model:
civitai_model = "/path/to/civitai_model.safetensors"
model = StableDiffusion(img_height=512, img_width=512, jit_compile=True, clip_skip=-2, civitai_model=civitai_model)
img = model.text_to_image(
    "a cute girl.",
    num_steps=25,
    seed=123456)
Image.fromarray(img[0]).save("out.jpg")

# for clip skip:
model = StableDiffusion(img_height=512, img_width=512, jit_compile=True, clip_skip=-2)
img = model.text_to_image(
    "a cute girl.",
    num_steps=25,
    seed=123456)
Image.fromarray(img[0]).save("out.jpg")

# for textual inversion:
model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
img = model.text_to_image(
    "a cute girl.",
    num_steps=25,
    seed=123456,
    embedding="/path/to/embedding.pt",
    negative_embedding="/path/to/negative_embedding.pt",
)
Image.fromarray(img[0]).save("out.jpg")

# for control net(canny mode)
model = StableDiffusion(img_height=512, img_width=512, jit_compile=True, clip_skip=-2,
                        controlnet_path="/path/to/control_sd15_canny.pth")
control_net_image = "/path/to/ref_image.jpg"
image = Image.open(control_net_image)
image = np.array(image)
canny = cv2.Canny(image, 100, 200)
canny = np.expand_dims(canny, axis=-1)
canny = np.concatenate([canny, canny, canny], axis=2)
Image.fromarray(canny).save("canny.jpg")
img = model.text_to_image(
    "a cute girl.",
    num_steps=25,
    seed=123456,
    control_net_image=np.expand_dims(canny, axis=0).astype(np.float32) / 255.0
)
```

* TODO
    - [x] Load Pytorch Weights
    - [x] Clip Skip
    - [x] Textual Inversion
    - [x] ControlNet
    - [ ] Long Prompt Weighting
    - [ ] Standard Lora

Distributed under the MIT License. See `LICENSE` for more information.

## Credits

Licenses for borrowed code can be found in following link:

- Stable Diffusion in TensorFlow / Keras - https://github.com/divamgupta/stable-diffusion-tensorflow
- Diffusion Bee (Stable Diffusion GUI App for MacOS) - https://github.com/divamgupta/diffusionbee-stable-diffusion-ui
- ControlNet - https://github.com/lllyasviel/ControlNet
- KerasCV - https://github.com/keras-team/keras-cv
- Diffusers - https://github.com/huggingface/diffusers

## Models

- ControlNet Models - https://huggingface.co/lllyasviel/ControlNet/tree/main/models
- Dreamlike Photoreal 2.0 - https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0
- Stable Diffusion v1.5 - https://huggingface.co/runwayml/stable-diffusion-v1-5
- Fine-Tuned VAE decoder https://huggingface.co/stabilityai/sd-vae-ft-mse

## Reach me on

- WeChat: DbgMonks
- QQ: 200759103
- E-Mail: gaozhihan@vip.qq.com

## Donating

If this project useful for you, please consider buying me a cup of coffee or sponsoring me!

<a href="https://paypal.me/cpuimage/USD10" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/black_img.png" alt="Buy Me A Coffee" style="height: auto !important;width: auto !important;" ></a>
