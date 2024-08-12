from diffusers import (
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from controlnet_aux import CannyDetector, NormalBaeDetector
from PIL import Image
import cv2

from transformers import pipeline
from torchvision.utils import save_image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class ControlNetSD(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="1.5",
        hf_key=None,
        t_range=[0.02, 0.80],
        control_type="normal",
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        self.control_type = control_type
        
        if self.control_type == "depth":
            controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
        elif self.control_type == "depth_pred":
            controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
            self.preprocessor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
            self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
            self.depth_model.to(device)
        elif self.control_type == "canny":
            controlnet_path = "lllyasviel/control_v11p_sd15_canny"
            self.preprocessor = CannyDetector()
        elif self.control_type == "normal":
            controlnet_path = "lllyasviel/control_v11p_sd15_normalbae"
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(device)
        else:
            print("Unsupported control type!!!")
            raise
        
        sd_path = "runwayml/stable-diffusion-v1-5"

        self.dtype = torch.float16 if fp16 else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=self.dtype,
        )

        # Create model
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path, controlnet=controlnet, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet

        self.scheduler = DDIMScheduler.from_pretrained(
            sd_path, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

        self.guidance_scale = 7.5
        self.condition_scale = 0.5
        self.canny_lower_bound = 1
        self.canny_upper_bound = 40

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
    
    @torch.no_grad()
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def forward_controlnet(self, latents, t, image_cond, condition_scale, encoder_hidden_states):
        return self.controlnet(
            latents.to(self.dtype),
            t.to(self.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.dtype),
            controlnet_cond=image_cond.to(self.dtype),
            conditioning_scale=condition_scale,
            return_dict=False,
        )

    @torch.no_grad()
    def forward_control_unet(self, latents, t, encoder_hidden_states, cross_attention_kwargs, down_block_additional_residuals, mid_block_additional_residual):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.dtype),
            t.to(self.dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.dtype),
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ).sample.to(input_dtype)
    
    @torch.no_grad()
    def prepare_image_cond(self, cond_rgb):
        if self.control_type == "depth":
            cond_depth = cond_rgb
            for i in range(cond_depth.shape[0]):
                cond_depth[i] = (cond_depth[i] - cond_depth[i].min()) / (cond_depth[i].max() - cond_depth[i].min())
            cond_depth = cond_depth.repeat(1, 3, 1, 1)
            cond_depth = F.interpolate(
                cond_depth, (512, 512), mode="bilinear", align_corners=False
            )
            control = cond_depth.detach()
        elif self.control_type == "depth_pred":
            control = []
            tmp_dir = "/mnt/workspace/tmp.jpg"
            for i in range(cond_rgb.shape[0]):
                save_image(cond_rgb[i], tmp_dir)
                image = Image.open(tmp_dir)
                inputs = self.preprocessor(images=image, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.depth_model(**inputs)
                    predicted_depth = outputs.predicted_depth
                control_ = predicted_depth.unsqueeze(0).repeat(1, 3, 1, 1)
                control_ = (control_ - control_.min()) / (control_.max() - control_.min())
                control_ = F.interpolate(
                    control_, (512, 512), mode="bilinear", align_corners=False
                )
                control.append(control_)
            control = torch.cat(control, dim=0)
        elif self.control_type == "canny":
            control = []
            for i in range(cond_rgb.shape[0]):
                canny = (
                    (cond_rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy()
                )
                blurred_img = cv2.blur(canny, ksize=(5, 5))
                detected_map = self.preprocessor(
                    blurred_img, self.canny_lower_bound, self.canny_upper_bound
                )
                control_ = (
                    torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
                )
                # control_ = control_.unsqueeze(-1).repeat(1, 1, 3)
                control_ = control_.unsqueeze(0)
                control_ = control_.permute(0, 3, 1, 2)
                control_ = F.interpolate(
                    control_, (512, 512), mode="bilinear", align_corners=False
                )
                control.append(control_)
            control = torch.cat(control, dim=0)
        elif self.control_type == "normal":
            control = []
            for i in range(cond_rgb.shape[0]):
                cond_rgb_ = (
                    (cond_rgb[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy()
                )
                detected_map = self.preprocessor(cond_rgb_)
                control_ = (
                    torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
                )
                control_ = control_.unsqueeze(0)
                control_ = control_.permute(0, 3, 1, 2)
                control_ = F.interpolate(
                    control_, (512, 512), mode="bilinear", align_corners=False
                )
                control.append(control_)
            control = torch.cat(control, dim=0)
        return control
    
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        image_cond,
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        vers=None, hors=None,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if hors is None:
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'
                
                # embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])
                
                text_z = []
                for h in hors:
                    if h >= -90 and h < 90:
                        if h >= 0:
                            r = 1 - h / 90
                        else:
                            r = 1 + h / 90
                        start_z = self.embeddings['front']
                        end_z = self.embeddings['side']
                    else:
                        if h >= 0 :
                            r = 1 - (h - 90) / 90
                        else:
                            r = 1 + (h + 90) / 90
                        start_z = self.embeddings['side']
                        end_z = self.embeddings['back']
                    text_z.append(r * start_z + (1 - r) * end_z)
                
                embeddings = torch.cat(text_z + [self.embeddings['neg'].expand(batch_size, -1, -1)])

            image_cond = self.prepare_image_cond(image_cond)
            
            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=embeddings,
                image_cond=image_cond.repeat(2,1,1,1),
                condition_scale=self.condition_scale,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                tt,
                encoder_hidden_states=embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        cond_depth,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            # noise_pred = self.unet(
            #     latent_model_input, t, encoder_hidden_states=embeddings
            # ).sample

            down_block_res_samples, mid_block_res_sample = self.forward_controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=embeddings,
                image_cond=cond_depth,
                condition_scale=1.0,
            )

            noise_pred = self.forward_control_unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeddings,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond 
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        cond_depth,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)

        # Text embeds -> img latents
        latents = self.produce_latents(
            cond_depth=cond_depth,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--depth_img", default="/mnt/workspace/image.png", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="1.5",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    # seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = ControlNetSD(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    print(f'[INFO] loading depth image from {opt.depth_img} ...')
    depth_img = kiui.read_image(opt.depth_img, mode='tensor')
    depth_img = depth_img[...,None].permute(2, 0, 1).repeat(3,1,1).unsqueeze(0).contiguous().to(device)
    depth_img = F.interpolate(depth_img, (512, 512), mode='bilinear', align_corners=False)

    imgs = sd.prompt_to_img(opt.prompt, depth_img, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()