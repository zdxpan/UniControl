'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
'''
import sys

sys.path.append('./')
from share import *
import random
save_memory = False

import cv2
import einops
import gradio as gr
import numpy as np
import torch
from PIL import Image
from annotator.util import resize_image, HWC3
from pytorch_lightning import seed_everything
from annotator.uniformer_base import UniformerDetector
from annotator.hed import HEDdetector
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.outpainting import Outpainter
from annotator.openpose import OpenposeDetector
from annotator.inpainting import Inpainter
from annotator.grayscale import GrayscaleConverter
from annotator.blur import Blurrer
import cvlib as cv

apply_uniformer = UniformerDetector()
apply_midas = MidasDetector()
apply_canny = CannyDetector()
apply_hed = HEDdetector()
model_outpainting = Outpainter()
apply_openpose = OpenposeDetector()
model_grayscale = GrayscaleConverter()
model_blur = Blurrer()
model_inpainting = Inpainter()


from cldm.model import create_model, load_state_dict
from cldm.ddim_unicontrol_hacked import DDIMSampler


model = create_model('./models/cldm_v15_unicontrol.yaml').cpu()
model.load_state_dict(load_state_dict('./ckpts/unicontrol.ckpt', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

task_to_name = {'hed': 'control_hed', 'canny': 'control_canny', 'seg': 'control_seg', 'segbase': 'control_seg', 'depth': 'control_depth', 'normal': 'control_normal', 'openpose': 'control_openpose', 'bbox': 'control_bbox', 'grayscale': 'control_grayscale', 'outpainting': 'control_outpainting', 'hedsketch': 'control_hedsketch'}

name_to_instruction = {"control_hed": "hed map to image", "control_canny": "canny edge to image", "control_seg": "segmentation map to image", "control_depth": "depth map to image", "control_normal": "normal surface map to image", "control_img": "image editing", "control_openpose": "human pose skeleton to image", "control_hedsketch": "sketch to image", "control_bbox": "bounding box to image", "control_outpainting": "image outpainting"}

def midas(img, res):
    img = resize_image(HWC3(img), res)
    results = apply_midas(img)
    return results

def outpainting(img, res, rand_h, rand_w):
    img = resize_image(HWC3(img), res)   
    result = model_outpainting(img, rand_h, rand_w)
    return result

def grayscale(img, res):
    img = resize_image(HWC3(img), res)
    result = model_grayscale(img)
    return result

def blur(img, res, ksize):
    img = resize_image(HWC3(img), res)   
    result = model_blur(img, ksize)
    return result

def inpainting(img, res,  rand_h, rand_h_1, rand_w, rand_w_1):
    img = resize_image(HWC3(img), res)   
    result = model_inpainting(img,  rand_h, rand_h_1, rand_w, rand_w_1)
    return result


def get_normal(input_image, num_samples, image_resolution, detect_resolution):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    _, detected_map = apply_midas(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return detected_map, control

def get_pose(input_image, num_samples, image_resolution, detect_resolution):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return detected_map, control

def get_sketch(input_image, num_samples, image_resolution, detect_resolution):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map = apply_hed(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    # sketch the hed image
    retry = 0
    cnt = 0
    while retry == 0:
        threshold_value = np.random.randint(110, 160)
        kernel_size = 3
        alpha = 1.5
        beta = 50
        binary_image = cv2.threshold(detected_map, threshold_value, 255, cv2.THRESH_BINARY)[1]
        inverted_image = cv2.bitwise_not(binary_image)
        smoothed_image = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 0)
        sketch_image = cv2.convertScaleAbs(smoothed_image, alpha=alpha, beta=beta)
        if np.sum(sketch_image < 5) > 0.005 * sketch_image.shape[0] * sketch_image.shape[1] or cnt == 5:
            retry = 1
        else:
            cnt += 1
    detected_map = sketch_image
    detected_map =  cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) 
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return detected_map, control

def get_colorization(input_image, num_samples, image_resolution):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map = grayscale(input_image, image_resolution)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = detected_map[:,:,np.newaxis]
    detected_map = detected_map.repeat(3, axis=2)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return detected_map, control

def get_deblur(input_image, num_samples, image_resolution, ksize):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map = blur(input_image, image_resolution, ksize)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    return detected_map, control

def get_canny(input_image, num_samples, image_resolution, low_threshold, high_threshold):
    # input_image = Image.fromarray(input_image)
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape
    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return 255 - detected_map, control

def get_hed(input_image, num_samples, image_resolution, detect_resolution):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map = apply_hed(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return detected_map, control

def get_depth(input_image, num_samples,image_resolution, detect_resolution):
    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return detected_map, control

def get_seg(input_image, num_samples, image_resolution, detect_resolution):

    input_image = HWC3(input_image)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    return detected_map, control

def pre_process(input_image1, input_image2, input_image3, control1, control2, control3, image_resolution, detect_resolution, 
                low_threshold, high_threshold,
               prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, scale1, scale2, scale3, TASKS_SWITCH):
    with torch.no_grad():
        tasks = [control1, control2, control3]
        tasks = [i for i in tasks if i != ""]
        if len(tasks) < 1:
            return None
        res = []
        controls = []
        def get_fixed_seed(seed):
            if seed is None or seed == '' or seed == -1:
                return int(random.randrange(2147483647))

            return seed
        seed = get_fixed_seed(seed)
        print("seed is :", seed)
        # random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        if input_image1 is not None:
            img = resize_image(HWC3(input_image1), image_resolution)
        elif input_image2 is not None:
            img = resize_image(HWC3(input_image1), image_resolution)
        else:
            return None
        H, W, C = img.shape
        task_instruction = ""
        task_scales = []
        all_task_feature = []
        for task, image, sub_scale in zip(tasks, [input_image1, input_image2, input_image3], [scale1, scale2, scale3]):
            if  image is None or task is None or task not in new_task_to_name:
                continue
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LANCZOS4)
            # image = resize_image(HWC3(image), image_resolution)
            if task == "canny":
                sub_res, control = get_canny(np.asarray(image), num_samples, image_resolution, low_threshold, high_threshold)
            elif task == "seg":
                sub_res, control = get_seg(np.asarray(image), num_samples, image_resolution, detect_resolution)
            elif task == "depth":
                sub_res, control = get_depth(np.asarray(image), num_samples, image_resolution, detect_resolution)
            elif task == "hed":
                sub_res, control = get_hed(np.asarray(image), num_samples, image_resolution, detect_resolution)
            elif task == "openpose":
                sub_res, control = get_pose(np.asarray(image), num_samples, image_resolution, detect_resolution)
            else:
                continue
            taskname = task_to_name[task]
            task_instruct = name_to_instruction[taskname]
            if len(task_instruction) > 2:
                task_instruction = "and" + task_instruct
            else:
                task_instruction = task_instruct
            res.append(Image.fromarray(sub_res))
            print("control.shape", control.shape)
            controls.append(control.cuda())
            task_scales.append(sub_scale)
            all_task_feature.append(model.get_learned_conditioning(task_instruct)[:,:1,:])
            # task = task

        if len(controls) == 0:
            return None
        all_tasks = [task_to_name[i]  for i in tasks]
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        if len(controls) > 1:
            task_dic["all_tasks"] = all_tasks
            task_dic["task_scale"] = task_scales
            task_dic["split_switch"] = TASKS_SWITCH
            task_dic["all_task_feature"] = all_task_feature
        cond = {"c_concat": controls, "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
        un_cond = {"c_concat": None if guess_mode else controls, "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)
            
    #         model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [res + results, seed]

new_task_to_name = {'hed': 'control_hed', 'canny': 'control_canny', 'seg': 'control_seg', 'segbase': 'control_seg', 'depth': 'control_depth', 'openpose': 'control_openpose'}
control_keys = list(new_task_to_name.keys())
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## UniControl Stable Diffusion with multil or singel Maps")
    with gr.Row():
        input_image1 = gr.Image(source='upload', type="numpy")
        input_image2 = gr.Image(source='upload', type="numpy")
        input_image3 = gr.Image(source='upload', type="numpy")
    with gr.Row():
        control1 = gr.Dropdown(choices = control_keys, label = "control1")
        control2 = gr.Dropdown(choices = control_keys, label = "control2")
        control3 = gr.Dropdown(choices = control_keys, label = "control3")
    with gr.Row():
        scale1 = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=10.0, value=0.5, step=0.1)
        scale2 = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=10.0, value=0.5, step=0.1)
        scale3 = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=10.0, value=0.5, step=0.1)
    with gr.Row():
        with gr.Column():
            # input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            pre_button = gr.Button(label="Preview")
            # run_button = gr.Button(label="Run")
            TASKS_SWITCH = gr.Slider(label="split Calculate", minimum=0, maximum=1, value=0, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                # scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                # seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='bad face, low quality')

        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    # ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution,
            # ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
    # run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    new_ips = [input_image1, input_image2, input_image3, control1, control2, control3, image_resolution, detect_resolution, low_threshold, high_threshold,
               prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, scale1, scale2, scale3, TASKS_SWITCH
               ]
    pre_button.click(fn=pre_process, inputs=new_ips, outputs=[result_gallery, seed])


block.launch(share=False, server_name='0.0.0.0')
