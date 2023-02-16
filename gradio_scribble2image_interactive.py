# This file is adapted from https://github.com/lllyasviel/ControlNet/blob/f4748e3630d8141d7765e2bd9b1e348f47847707/gradio_scribble2image_interactive.py
# The original license file is LICENSE.ControlNet in this repo.
import gradio as gr
import numpy as np


def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255


def create_demo(process, max_images=12):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                '## Control Stable Diffusion with Interactive Scribbles')
        with gr.Row():
            with gr.Column():
                canvas_width = gr.Slider(label='Canvas Width',
                                         minimum=256,
                                         maximum=1024,
                                         value=512,
                                         step=1)
                canvas_height = gr.Slider(label='Canvas Height',
                                          minimum=256,
                                          maximum=1024,
                                          value=512,
                                          step=1)
                create_button = gr.Button(label='Start',
                                          value='Open drawing canvas!')
                input_image = gr.Image(source='upload',
                                       type='numpy',
                                       tool='sketch')
                gr.Markdown(
                    value=
                    'Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
                    'Just click on the small pencil icon in the upper right corner of the above block.'
                )
                create_button.click(fn=create_canvas,
                                    inputs=[canvas_width, canvas_height],
                                    outputs=[input_image],
                                    queue=False)
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    num_samples = gr.Slider(label='Images',
                                            minimum=1,
                                            maximum=max_images,
                                            value=1,
                                            step=1)
                    image_resolution = gr.Slider(label='Image Resolution',
                                                 minimum=256,
                                                 maximum=768,
                                                 value=512,
                                                 step=256)
                    ddim_steps = gr.Slider(label='Steps',
                                           minimum=1,
                                           maximum=100,
                                           value=20,
                                           step=1)
                    scale = gr.Slider(label='Guidance Scale',
                                      minimum=0.1,
                                      maximum=30.0,
                                      value=9.0,
                                      step=0.1)
                    seed = gr.Slider(label='Seed',
                                     minimum=-1,
                                     maximum=2147483647,
                                     step=1,
                                     randomize=True)
                    eta = gr.Number(label='eta (DDIM)', value=0.0)
                    a_prompt = gr.Textbox(
                        label='Added Prompt',
                        value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(
                        label='Negative Prompt',
                        value=
                        'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
                    )
            with gr.Column():
                result_gallery = gr.Gallery(label='Output',
                                            show_label=False,
                                            elem_id='gallery').style(
                                                grid=2, height='auto')
        ips = [
            input_image, prompt, a_prompt, n_prompt, num_samples,
            image_resolution, ddim_steps, scale, seed, eta
        ]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    return demo
