# This file is adapted from https://github.com/lllyasviel/ControlNet/blob/f4748e3630d8141d7765e2bd9b1e348f47847707/gradio_canny2image.py
# The original license file is LICENSE.ControlNet in this repo.
import gradio as gr


def create_demo(process, max_images=12):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Control Stable Diffusion with Canny Edge Maps')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='numpy')
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
                    low_threshold = gr.Slider(label='Canny low threshold',
                                              minimum=1,
                                              maximum=255,
                                              value=100,
                                              step=1)
                    high_threshold = gr.Slider(label='Canny high threshold',
                                               minimum=1,
                                               maximum=255,
                                               value=200,
                                               step=1)
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
            image_resolution, ddim_steps, scale, seed, eta, low_threshold,
            high_threshold
        ]
        run_button.click(fn=process,
                         inputs=ips,
                         outputs=[result_gallery],
                         api_name='canny')
    return demo
