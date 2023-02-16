# This file is adapted from gradio_*.py in https://github.com/lllyasviel/ControlNet/tree/f4748e3630d8141d7765e2bd9b1e348f47847707
# The original license file is LICENSE.ControlNet in this repo.
from __future__ import annotations

import pathlib
import random
import shlex
import subprocess
import sys

import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

sys.path.append('ControlNet')

import config
from annotator.canny import apply_canny
from annotator.hed import apply_hed, nms
from annotator.midas import apply_midas
from annotator.mlsd import apply_mlsd
from annotator.openpose import apply_openpose
from annotator.uniformer import apply_uniformer
from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from share import *

ORIGINAL_MODEL_NAMES = {
    'canny': 'control_sd15_canny.pth',
    'hough': 'control_sd15_mlsd.pth',
    'hed': 'control_sd15_hed.pth',
    'scribble': 'control_sd15_scribble.pth',
    'pose': 'control_sd15_openpose.pth',
    'seg': 'control_sd15_seg.pth',
    'depth': 'control_sd15_depth.pth',
    'normal': 'control_sd15_normal.pth',
}
ORIGINAL_WEIGHT_ROOT = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/'

LIGHTWEIGHT_MODEL_NAMES = {
    'canny': 'control_canny-fp16.safetensors',
    'hough': 'control_mlsd-fp16.safetensors',
    'hed': 'control_hed-fp16.safetensors',
    'scribble': 'control_scribble-fp16.safetensors',
    'pose': 'control_openpose-fp16.safetensors',
    'seg': 'control_seg-fp16.safetensors',
    'depth': 'control_depth-fp16.safetensors',
    'normal': 'control_normal-fp16.safetensors',
}
LIGHTWEIGHT_WEIGHT_ROOT = 'https://huggingface.co/webui/ControlNet-modules-safetensors/resolve/main/'


class Model:
    def __init__(self,
                 model_config_path: str = 'ControlNet/models/cldm_v15.yaml',
                 model_dir: str = 'models',
                 use_lightweight: bool = True):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(model_config_path).to(self.device)
        self.ddim_sampler = DDIMSampler(self.model)
        self.task_name = ''

        self.model_dir = pathlib.Path(model_dir)

        self.use_lightweight = use_lightweight
        if use_lightweight:
            self.model_names = LIGHTWEIGHT_MODEL_NAMES
            self.weight_root = LIGHTWEIGHT_WEIGHT_ROOT
            base_model_url = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
            self.load_base_model(base_model_url)
        else:
            self.model_names = ORIGINAL_MODEL_NAMES
            self.weight_root = ORIGINAL_WEIGHT_ROOT
        self.download_models()

    def download_base_model(self, model_url: str) -> pathlib.Path:
        model_name = model_url.split('/')[-1]
        out_path = self.model_dir / model_name
        if not out_path.exists():
            subprocess.run(shlex.split(f'wget {model_url} -O {out_path}'))
        return out_path

    def load_base_model(self, model_url: str) -> None:
        model_path = self.download_base_model(model_url)
        self.model.load_state_dict(load_state_dict(model_path,
                                                   location=self.device.type),
                                   strict=False)

    def load_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        weight_path = self.get_weight_path(task_name)
        if not self.use_lightweight:
            self.model.load_state_dict(
                load_state_dict(weight_path, location=self.device))
        else:
            self.model.control_model.load_state_dict(
                load_state_dict(weight_path, location=self.device.type))
        self.task_name = task_name

    def get_weight_path(self, task_name: str) -> str:
        if 'scribble' in task_name:
            task_name = 'scribble'
        return f'{self.model_dir}/{self.model_names[task_name]}'

    def download_models(self) -> None:
        self.model_dir.mkdir(exist_ok=True, parents=True)
        for name in self.model_names.values():
            out_path = self.model_dir / name
            if out_path.exists():
                continue
            subprocess.run(
                shlex.split(f'wget {self.weight_root}{name} -O {out_path}'))

    @torch.inference_mode()
    def process_canny(self, input_image, prompt, a_prompt, n_prompt,
                      num_samples, image_resolution, ddim_steps, scale, seed,
                      eta, low_threshold, high_threshold):
        self.load_weight('canny')

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [255 - detected_map] + results

    @torch.inference_mode()
    def process_hough(self, input_image, prompt, a_prompt, n_prompt,
                      num_samples, image_resolution, detect_resolution,
                      ddim_steps, scale, seed, eta, value_threshold,
                      distance_threshold):
        self.load_weight('hough')

        input_image = HWC3(input_image)
        detected_map = apply_mlsd(resize_image(input_image, detect_resolution),
                                  value_threshold, distance_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [
            255 - cv2.dilate(detected_map,
                             np.ones(shape=(3, 3), dtype=np.uint8),
                             iterations=1)
        ] + results

    @torch.inference_mode()
    def process_hed(self, input_image, prompt, a_prompt, n_prompt, num_samples,
                    image_resolution, detect_resolution, ddim_steps, scale,
                    seed, eta):
        self.load_weight('hed')

        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results

    @torch.inference_mode()
    def process_scribble(self, input_image, prompt, a_prompt, n_prompt,
                         num_samples, image_resolution, ddim_steps, scale,
                         seed, eta):
        self.load_weight('scribble')

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [255 - detected_map] + results

    @torch.inference_mode()
    def process_scribble_interactive(self, input_image, prompt, a_prompt,
                                     n_prompt, num_samples, image_resolution,
                                     ddim_steps, scale, seed, eta):
        self.load_weight('scribble')

        img = resize_image(HWC3(input_image['mask'][:, :, 0]),
                           image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) > 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [255 - detected_map] + results

    @torch.inference_mode()
    def process_fake_scribble(self, input_image, prompt, a_prompt, n_prompt,
                              num_samples, image_resolution, detect_resolution,
                              ddim_steps, scale, seed, eta):
        self.load_weight('scribble')

        input_image = HWC3(input_image)
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [255 - detected_map] + results

    @torch.inference_mode()
    def process_pose(self, input_image, prompt, a_prompt, n_prompt,
                     num_samples, image_resolution, detect_resolution,
                     ddim_steps, scale, seed, eta):
        self.load_weight('pose')

        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(
            resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results

    @torch.inference_mode()
    def process_seg(self, input_image, prompt, a_prompt, n_prompt, num_samples,
                    image_resolution, detect_resolution, ddim_steps, scale,
                    seed, eta):
        self.load_weight('seg')

        input_image = HWC3(input_image)
        detected_map = apply_uniformer(
            resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results

    @torch.inference_mode()
    def process_depth(self, input_image, prompt, a_prompt, n_prompt,
                      num_samples, image_resolution, detect_resolution,
                      ddim_steps, scale, seed, eta):
        self.load_weight('depth')

        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(
            resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results

    @torch.inference_mode()
    def process_normal(self, input_image, prompt, a_prompt, n_prompt,
                       num_samples, image_resolution, detect_resolution,
                       ddim_steps, scale, seed, eta, bg_threshold):
        self.load_weight('normal')

        input_image = HWC3(input_image)
        _, detected_map = apply_midas(resize_image(input_image,
                                                   detect_resolution),
                                      bg_th=bg_threshold)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H),
                                  interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(
            detected_map[:, :, ::-1].copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat': [control],
            'c_crossattn': [
                self.model.get_learned_conditioning(
                    [prompt + ', ' + a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [self.model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        samples, intermediates = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond)

        if config.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [detected_map] + results
