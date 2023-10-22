import copy
import datetime
import gc
import os
import re
from typing import Literal, Optional

import numpy as np
import streamlit as st
from scipy.ndimage import correlate1d
from streamlit_drawable_canvas import st_canvas

from PIL import Image
from stable_diffusion.stable_diffusion import StableDiffusion

PIPELINE_NAMES = Literal["txt2img", "img2img", "inpaint"]
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
DEFAULT_PROMPT = "border collie puppy"
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"

global pipe


@st.cache_resource()
def get_pipeline(width, height):
    global pipe
    if pipe is None:
        pipe = StableDiffusion(img_height=height, img_width=width, jit_compile=True)
    else:
        if pipe.img_width != width or pipe.img_height != height:
            st.cache_resource.clear()
            del pipe
            gc.collect()
            pipe = StableDiffusion(img_height=height, img_width=width, jit_compile=True)
    return pipe


@st.cache_resource(max_entries=1)
def generate(
        prompt,
        pipeline_name: PIPELINE_NAMES,
        _image_input=None,
        _mask_input=None,
        negative_prompt=None,
        steps=50,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        guidance_scale=7.5,
        strength=1.0,
        seed=-1,
):
    pipe = get_pipeline(width=width, height=height)
    """Generates an image based on the given prompt and pipeline name"""
    negative_prompt = negative_prompt if negative_prompt else None
    p = st.progress(0)
    callback = lambda step: p.progress(step / strength / steps)
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=steps,
        callback=callback,
        seed=None if seed == -1 else seed,
        unconditional_guidance_scale=guidance_scale,
    )
    print("kwargs", kwargs)
    if pipeline_name == "inpaint" and _image_input and _mask_input:
        kwargs.update(reference_image=_image_input, inpaint_mask=_mask_input, reference_image_strength=strength)
        images = pipe.inpaint(**kwargs)
    elif pipeline_name == "txt2img":
        images = pipe.text_to_image(**kwargs)
    elif pipeline_name == "img2img" and _image_input:
        kwargs.update(
            reference_image=_image_input, reference_image_strength=strength)
        images = pipe.image_to_image(**kwargs)
    else:
        raise Exception(
            f"Cannot generate image for pipeline {pipeline_name} and {prompt}"
        )

    image = Image.fromarray(images[0])
    os.makedirs("outputs", exist_ok=True)
    filename = (
            "outputs/"
            + re.sub(r"\s+", "_", prompt)[:50]
            + f"_{datetime.datetime.now().timestamp()}"
    )
    image.save(f"{filename}.png")
    with open(f"{filename}.txt", "w") as f:
        f.write(f"Prompt: {prompt}\n\nNegative Prompt: {negative_prompt}")
    return image


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


def prompt_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    negative_prompt = st.text_area(
        "Negative prompt",
        value="",
        key=f"{prefix}-negative-prompt",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        steps = st.slider(
            "Number of inference steps",
            min_value=1,
            max_value=200,
            value=20,
            key=f"{prefix}-inference-steps",
        )
    with col2:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=0.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            key=f"{prefix}-guidance-scale",
        )
    with col3:
        seed = st.text_input(
            "seed",
            value=-1,
            key=f"{prefix}-seed",
        )
    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating image..."):
            image = generate(
                prompt,
                pipeline_name,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                **kwargs,
            )
            set_image(OUTPUT_IMAGE_KEY, image.copy())
            st.image(image)


def width_and_height_sliders(prefix):
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width",
            min_value=128,
            max_value=2048,
            step=64,
            value=DEFAULT_WIDTH,
            key=f"{prefix}-width",
        )
    with col2:
        height = st.slider(
            "Height",
            min_value=128,
            max_value=2048,
            step=64,
            value=DEFAULT_HEIGHT,
            key=f"{prefix}-height",
        )
    return width, height


def image_uploader(prefix):
    image = st.file_uploader("Image", ["jpg", "png"], key=f"{prefix}-uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        return image

    return get_image(LOADED_IMAGE_KEY)


def gaussian_blur(image, radius=3, h_axis=0, v_axis=1):
    def build_filter1d(kernel_size):
        if kernel_size == 1:
            filter1d = [1]
        else:
            triangle = [[1, 1]]
            for i in range(1, kernel_size - 1):
                cur_row = [1]
                prev_row = triangle[i - 1]
                for j in range(len(prev_row) - 1):
                    cur_row.append(prev_row[j] + prev_row[j + 1])
                cur_row.append(1)
                triangle.append(cur_row)
            filter1d = triangle[-1]
        filter1d = np.reshape(filter1d, (kernel_size,))
        return filter1d / np.sum(filter1d)

    weights = build_filter1d(radius)
    blurred_image = correlate1d(image, weights, axis=h_axis, output=None, mode="reflect", cval=0.0, origin=0)
    blurred_image = correlate1d(blurred_image, weights, axis=v_axis, output=None, mode="reflect", cval=0.0,
                                origin=0)
    return blurred_image


def txt2img_tab():
    prefix = "txt2img"
    width, height = width_and_height_sliders(prefix)
    prompt_and_generate_button(
        prefix, "txt2img", width=width, height=height,
    )


def img2img_tab():
    prefix = "img2img"
    col1, col2 = st.columns(2)

    with col1:
        image = image_uploader(prefix)
        if image:
            st.image(image)

    with col2:
        if image:
            strength = st.slider(
                "Strength (1.0 ignores the existing image so it's not a useful value)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key=f"{prefix}-strength",
            )
            prompt_and_generate_button(
                prefix, "img2img", _image_input=image, strength=strength
            )


def inpainting_tab():
    prefix = "inpaint"
    col1, col2 = st.columns(2)
    with col1:
        image_input, mask_input = None, None
        image = image_uploader(prefix)
        if image:
            feathering_strength = st.number_input("Feathering Strength", value=5, min_value=1, max_value=255)
            brush_size = st.number_input("Brush Size", value=50, min_value=1, max_value=100)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=brush_size,
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=image,
                update_streamlit=True,
                height=image.height,
                width=image.width,
                drawing_mode="freedraw",
                key=f"{prefix}-canvas")
            if not (not canvas_result or canvas_result.image_data is None):
                mask = canvas_result.image_data
                mask = np.asarray(mask[:, :, -1] > 0, np.uint8)
                if mask.sum() > 0:
                    if feathering_strength > 1:
                        mask = gaussian_blur(mask * 255, feathering_strength, h_axis=0, v_axis=1)
                    pil_mask = Image.fromarray(mask)
                    st.image(pil_mask)
                    image_input, mask_input = image, pil_mask
    with col2:
        if image_input and mask_input:
            strength = st.slider(
                "Strength of inpainting (1.0 essentially ignores the masked area of the original input image)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"{prefix}-strength",
            )
            prompt_and_generate_button(
                prefix,
                "inpaint",
                _image_input=image_input,
                _mask_input=mask_input,
                strength=strength,
            )


def main():
    global pipe
    pipe = None
    st.set_page_config(layout="wide")
    st.title("Stable Diffusion V1.5 Playground")
    tab1, tab2, tab3 = st.tabs(
        ["Text to Image (txt2img)", "Image to image (img2img)", "Inpainting"]
    )
    with tab1:
        txt2img_tab()

    with tab2:
        img2img_tab()

    with tab3:
        inpainting_tab()

    with st.sidebar:
        st.header("Latest Output Image")
        output_image = get_image(OUTPUT_IMAGE_KEY)
        if output_image:
            st.image(output_image)
            if st.button("Use this image for img2img and inpaint"):
                set_image(LOADED_IMAGE_KEY, copy.deepcopy(output_image))
                st.rerun()
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()
