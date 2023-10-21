import gradio as gr

from stable_diffusion.stable_diffusion import StableDiffusion


def inference_fn(prompt, negative_prompt, num_inference_steps, guidance_scale, seed, reference_image, denoise_strength):
    global SD_INSTANCE
    output = SD_INSTANCE.image_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=num_inference_steps,
        unconditional_guidance_scale=guidance_scale,
        reference_image=reference_image,
        reference_image_strength=denoise_strength,
        seed=None if seed == -1 else seed)
    return output[0]


def main():
    height = 512
    width = 512
    global SD_INSTANCE
    SD_INSTANCE = StableDiffusion(img_height=height, img_width=width, jit_compile=True)
    with gr.Blocks() as app:
        with gr.Tab("Image2Image"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Text Encoder")
                    prompt = gr.Textbox(label="prompt", value="hello stable diffusion")
                    negative_prompt = gr.Textbox(label="negative prompt", value="")
                    gr.Markdown("Sampler")
                    num_inference_steps = gr.Slider(label="steps", value=25, minimum=1, maximum=100, step=1,
                                                    interactive=True)
                    guidance_scale = gr.Slider(label="guidance scale", value=7.0, minimum=0.0, maximum=100.0, step=0.01,
                                               interactive=True)
                    seed = gr.Number(label='seed', value=-1, min_width=100, precision=0)
                    gr.Markdown("Image 2 Image")
                    denoise_strength = gr.Slider(label="denoise strength", value=0.8, minimum=0.0, maximum=1.0,
                                                 step=0.01,
                                                 interactive=True)
            with gr.Row():
                reference_image = gr.Image(width=width, height=height, label="Image 2 Image")
                output_image = gr.Image(width=width, height=height)
        inference_button = gr.Button("inference")
        inference_button.click(fn=inference_fn,
                               inputs=[prompt, negative_prompt, num_inference_steps,
                                       guidance_scale, seed, reference_image, denoise_strength],
                               outputs=output_image)

    app.launch()


if __name__ == '__main__':
    main()
