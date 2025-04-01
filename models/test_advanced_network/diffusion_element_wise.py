import torch
import numpy as np
from tqdm import tqdm
import clip
import math
from utils.utils import extract_floats_from_string
from utils.fading_channel_utils import l2_normalization
class DiffusionGenerator:
    def __init__(self, model, vae, text_embed,device, model_dtype=torch.float32):
        self.model = model
        self.vae = vae
        self.device = device
        self.model_dtype = model_dtype
        self.text_embed = text_embed
        from diffusers import DDPMScheduler
        self.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='scheduler')
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
    def encode_text(self,label, model):
        text_tokens = clip.tokenize(label, truncate=True).cuda()
        text_encoding = model.encode_text(text_tokens)
        return text_encoding.cpu()

    def sigmoid_np(self,x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_schedule(self,t, start=0, end=3, tau=0.7, clip_min=1e-6):

        v_start = self.sigmoid_np(start / tau)
        v_end = self.sigmoid_np(end / tau)
        output = self.sigmoid_np((t * (end - start) + start) / tau)
        output = (output - v_start) / (v_end - v_start)
        return output.clip(clip_min, 1 - clip_min)
    def sigmoid_schedule_inverse(self,output, start=0, end=3, tau=0.7, clip_min=1e-6):
        v_start = self.sigmoid_np(start / tau)
        v_end = self.sigmoid_np(end / tau)
        output = output * (v_end - v_start) + v_start
        t = (-start-tau*np.log(1/output-1))/(end-start)
        t = np.clip(t, start, end)
        return t
    def adjust_cfg_weight(self, cfg_weight, time_step,cfg_weighting_method):
        if cfg_weighting_method == 'constant':
            return cfg_weight
        elif 'pcs' in cfg_weighting_method:
            s = extract_floats_from_string(cfg_weighting_method)[0]
            w = (1 - np.cos(np.pi*(time_step)**s))/2 * cfg_weight
            return w
        elif 'clamp_linear' in cfg_weighting_method:
            s = extract_floats_from_string(cfg_weighting_method)[0]
            w = np.clip(time_step*2*cfg_weight,s,2*cfg_weight)
            return w
        elif cfg_weighting_method == 'sigmoid':
            return self.sigmoid_schedule(cfg_weight)
        else:
            raise ValueError(f'Unknown cfg_weighting_method: {cfg_weighting_method}')

    @torch.no_grad()
    def generate(self,
                 n_iter=30,
                 prompt=None, # embeddings to condition on
                 num_imgs=16,
                 class_guidance=3,
                 seed=10,  # for reproducibility
                 scale_factor=8,  # latent scaling before decoding - should be ~ std of latent space
                 img_size=32,  # height, width of latent
                 img_channel=4,  # number of channels in latent
                 sharp_f=0.1,
                 bright_f=0.1,
                 exponent=1,
                 seeds=None,
                 latent=None,
                 noise_levels=None,negative_prompt=None,diffusion_step=50,step_style='discrete',not_control=None,cfg_weighting_method='constant',
                 use_ddpm_plus=True, alphas_cumprod=None, curr_step=None,return_latent = False,c=None,controlnet=False,mask_step=0,mask_token=None,return_all_latent=False):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        class_guidance_ori = class_guidance
        labels = self.encode_text(prompt, self.text_embed)
        negative_labels = self.encode_text(negative_prompt, self.text_embed) if negative_prompt is not None else torch.zeros_like(labels)
        if step_style == 'discrete':
            if noise_levels is None:
                noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
            noise_levels[0] = 0.99
            timesteps = np.linspace(1, 981 if curr_step is None else curr_step, diffusion_step if curr_step>=diffusion_step else curr_step).astype(np.int32)
            noise_levels = torch.sqrt(1 - alphas_cumprod[timesteps]).tolist()
            noise_levels.reverse()
        elif step_style == 'continuous':
            curr_timestep = self.sigmoid_schedule_inverse(curr_step.cpu().numpy())
            timesteps = np.linspace(0.001,curr_timestep,diffusion_step)[:,:,0].transpose(1,0)
            noise_levels = np.sqrt(self.sigmoid_schedule(timesteps))
            #noise_levels = [math.sqrt(self.sigmoid_schedule(t)) for t in timesteps]
            noise_levels = np.flip(noise_levels,axis=1)
            timesteps = np.flip(timesteps,axis=1)

        if use_ddpm_plus:
            #lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]# log snr
            lambdas = np.log((1 - noise_levels) / noise_levels)
            hs = []
            for i in range(lambdas.shape[1]):
                hs.append(lambdas[:,i] - lambdas[:,i-1])
            hs = np.stack(hs,axis=1)
            rs = []
            for i in range(hs.shape[1]):
                rs.append(hs[:,i-1] / hs[:,i])
            rs = np.stack(rs,axis=1)

            #lambdas = np.log((1-noise_levels)/noise_levels)

        if latent is None:
            x_t = self.initialize_image(seeds, num_imgs, img_size, seed, img_channel)
        else:
            x_t = latent

        labels = torch.cat([labels, negative_labels])
        self.model.eval()
        if not_control is None:
            not_control = [1 for i in range(latent.size(0))] + [1 for i in range(latent.size(0))]
        x0_pred_prev = None
        if return_all_latent:
            all_latent = [x_t.detach().clone().cpu()]
        for i in tqdm(range(noise_levels.shape[1] - 1)):
            curr_noise, next_noise = noise_levels[:,i], noise_levels[:,i + 1]
            curr_timestep,next_timestep = timesteps[:,i],timesteps[:,i+1]
            class_guidance = self.adjust_cfg_weight(class_guidance_ori,curr_timestep,cfg_weighting_method)
            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance,c,controlnet,mask_token if mask_step>0 else None,not_control=not_control)
            mask_step =mask_step -1 if mask_step>0 else 0
            x_pred_scale = np.sqrt(1 - next_noise ** 2) - next_noise / curr_noise * np.sqrt(1 - curr_noise ** 2)
            x_t_scale = next_noise / curr_noise
            # x_pred_scale = math.sqrt(next_noise) - math.sqrt(curr_noise * (1 - next_noise) / (1 - curr_noise))
            # x_t_scale = math.sqrt((1 - next_noise) / (1 - curr_noise))
            x_pred_scale = self.expand_scalar(torch.from_numpy(x_pred_scale).to(self.device), x_t.shape)
            x_t_scale = self.expand_scalar(torch.from_numpy(x_t_scale).to(self.device), x_t.shape)
            if x0_pred_prev is None:
                x_t = (x_pred_scale * x0_pred + x_t_scale * x_t)
                # x_t = (math.sqrt(curr_noise - next_noise) * x0_pred + math.sqrt(next_noise) * x_t) / math.sqrt(curr_noise)
            else:
                if use_ddpm_plus:
                    rs_i_1 = self.expand_scalar(torch.from_numpy(rs[:,i - 1])[:,None].to(self.device),x_t.shape)
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs_i_1)) * x0_pred - (1 / (2 * rs_i_1)) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = x_pred_scale * D + x_t_scale * x_t
                # x_t = (math.sqrt(curr_noise - next_noise) * D + math.sqrt(next_noise) * x_t) / math.sqrt(curr_noise)
            if return_all_latent:
                all_latent.append(x_t.detach().clone().cpu())
            x0_pred_prev = x0_pred


        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance,c,controlnet,not_control=not_control)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
        if return_all_latent:
            return x0_pred_img, all_latent
        return x0_pred_img, x0_pred
    def generate_noisy_latent(self,latent,cur_noise_level,signal_scale,noise_scale):
        m = signal_scale/(np.sqrt(1-cur_noise_level**2))
        c = cur_noise_level - noise_scale/m
        c = torch.clamp(c,0,1)
        x_t = latent/m + c * torch.randn_like(latent)
        x_t = l2_normalization(x_t)
        return x_t
    def update_x_t_with_latent(self,latent,x_0_pred,cur_noise_level,next_noise_level,signal_scale,noise_scale):
        update_mask = (noise_scale>=next_noise_level).to(torch.float32)

        x_pred_scale = math.sqrt(1 - next_noise_level ** 2) - next_noise_level / noise_scale * torch.sqrt(1 - noise_scale ** 2)
        x_t_scale = next_noise_level / noise_scale
        g_s = (x_pred_scale * x_0_pred + x_t_scale * latent)
        a_s = torch.ones_like(signal_scale)* math.sqrt(1 - next_noise_level ** 2)
        b_s = next_noise_level

        new_latent = g_s * update_mask + latent * (1 - update_mask)
        new_signal_scale = a_s * update_mask + signal_scale * (1 - update_mask)
        new_noise_scale = b_s * update_mask + noise_scale * (1 - update_mask)
        return new_latent,new_signal_scale,new_noise_scale

    @torch.no_grad()
    def generate_fading(self,
                 n_iter=30,
                 prompt=None, # embeddings to condition on
                 num_imgs=16,
                 class_guidance=3,
                 seed=10,  # for reproducibility
                 scale_factor=8,  # latent scaling before decoding - should be ~ std of latent space
                 img_size=32,  # height, width of latent
                 img_channel=4,  # number of channels in latent
                 sharp_f=0.1,
                 bright_f=0.1,
                 exponent=1,
                 seeds=None,
                 latent=None,
                signal_scale=1,noise_scale=0,signal=None,noise=None,
                 noise_levels=None,negative_prompt=None,diffusion_step=50,step_style='discrete',not_control=None,cfg_weighting_method='constant',
                 use_ddpm_plus=True, alphas_cumprod=None, curr_step=None,return_latent = False,c=None,controlnet=False,mask_step=0,mask_token=None,return_all_latent=False):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        class_guidance_ori = class_guidance
        labels = self.encode_text(prompt, self.text_embed)
        negative_labels = self.encode_text(negative_prompt, self.text_embed) if negative_prompt is not None else torch.zeros_like(labels)
        if step_style == 'discrete':
            if noise_levels is None:
                noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
            noise_levels[0] = 0.99
            timesteps = np.linspace(1, 981 if curr_step is None else curr_step, diffusion_step if curr_step>=diffusion_step else curr_step).astype(np.int32)
            noise_levels = torch.sqrt(1 - alphas_cumprod[timesteps]).tolist()
            noise_levels.reverse()
        elif step_style == 'continuous':
            curr_timestep = self.sigmoid_schedule_inverse(curr_step)
            timesteps = np.linspace(0.001,curr_timestep,diffusion_step).tolist()
            noise_levels = [math.sqrt(self.sigmoid_schedule(t)) for t in timesteps]
            noise_levels.reverse()
            timesteps.reverse()

        #we first limit the noise scale to be smaller than noise_levels[0]
        noise_scale = torch.clamp(noise_scale,0,noise_levels[0])
        signal_scale = torch.sqrt(1-noise_scale**2)
        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        if latent is None:
            x_t = self.initialize_image(seeds, num_imgs, img_size, seed, img_channel)
        else:
            x_t = latent

        labels = torch.cat([labels, negative_labels])
        self.model.eval()
        if not_control is None:
            not_control = [1 for i in range(latent.size(0))] + [1 for i in range(latent.size(0))]
        x0_pred_prev = None

        if return_all_latent:
            all_latent = [x_t.detach().clone().cpu()]
        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            curr_timestep,next_timestep = timesteps[i],timesteps[i+1]
            class_guidance = self.adjust_cfg_weight(class_guidance_ori,curr_timestep,cfg_weighting_method)
            x_t = self.generate_noisy_latent(latent,curr_noise,signal_scale,noise_scale)
            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance,c,controlnet,mask_token if mask_step>0 else None,not_control=not_control)
            mask_step =mask_step -1 if mask_step>0 else 0
            latent,signal_scale,noise_scale = self.update_x_t_with_latent(latent,x0_pred,curr_noise,next_noise,signal_scale,noise_scale)
            x_t = latent
            if return_all_latent:
                all_latent.append(x_t.detach().clone().cpu())
            x0_pred_prev = x0_pred


        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance,c,controlnet,not_control=not_control)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
        if return_all_latent:
            return x0_pred_img, all_latent
        return x0_pred_img, x0_pred
    @torch.no_grad()
    def generate_no_bar(self,
                 n_iter=30,
                 prompt=None, # embeddings to condition on
                 num_imgs=16,
                 class_guidance=3,
                 seed=10,  # for reproducibility
                 scale_factor=8,  # latent scaling before decoding - should be ~ std of latent space
                 img_size=32,  # height, width of latent
                 img_channel=4,  # number of channels in latent
                 sharp_f=0.1,
                 bright_f=0.1,
                 exponent=1,
                 seeds=None,
                 latent=None,
                 noise_levels=None,
                 use_ddpm_plus=True, alphas_cumprod=None, curr_step=None,return_latent = False,c=None,controlnet=False,mask_step=0,mask_token=None,return_all_latent=False):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        labels = self.encode_text(prompt, self.text_embed)
        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99
        timesteps = np.linspace(1, 981 if curr_step is None else curr_step, 50 if curr_step>=50 else curr_step).astype(np.int32)
        noise_levels = torch.sqrt(1 - alphas_cumprod[timesteps]).tolist()
        noise_levels.reverse()
        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        if latent is None:
            x_t = self.initialize_image(seeds, num_imgs, img_size, seed, img_channel)
        else:
            x_t = latent

        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        x0_pred_prev = None
        if return_all_latent:
            all_latent = [x_t.detach().clone().cpu()]
        #for i in tqdm(range(len(noise_levels) - 1)):
        for i in range(len(noise_levels) - 1):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance,c,controlnet,mask_token if mask_step>0 else None)
            mask_step =mask_step -1 if mask_step>0 else 0
            x_pred_scale = math.sqrt(1 - next_noise ** 2) - next_noise / curr_noise * math.sqrt(1 - curr_noise ** 2)
            x_t_scale = next_noise / curr_noise
            # x_pred_scale = math.sqrt(next_noise) - math.sqrt(curr_noise * (1 - next_noise) / (1 - curr_noise))
            # x_t_scale = math.sqrt((1 - next_noise) / (1 - curr_noise))
            if x0_pred_prev is None:
                x_t = (x_pred_scale * x0_pred + x_t_scale * x_t)
                # x_t = (math.sqrt(curr_noise - next_noise) * x0_pred + math.sqrt(next_noise) * x_t) / math.sqrt(curr_noise)
            else:
                if use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = x_pred_scale * D + x_t_scale * x_t
                # x_t = (math.sqrt(curr_noise - next_noise) * D + math.sqrt(next_noise) * x_t) / math.sqrt(curr_noise)
            if return_all_latent:
                all_latent.append(x_t.detach().clone().cpu())
            x0_pred_prev = x0_pred


        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance,c,controlnet)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
        if return_all_latent:
            return x0_pred_img, all_latent
        return x0_pred_img, x0_pred
    def __call__(self,
                 n_iter=30,
                 prompt=None,  # embeddings to condition on
                 num_imgs=16,
                 cur_noise_level=1,
                 class_guidance=3,
                 seed=10,  # for reproducibility
                 scale_factor=8,  # latent scaling before decoding - should be ~ std of latent space
                 img_size=32,  # height, width of latent
                 img_channel=4,  # number of channels in latent
                 sharp_f=0.1,
                 bright_f=0.1,
                 exponent=1,
                 seeds=None,
                 latent = None,
                 noise_levels=None,
                 use_ddpm_plus=False,
                 return_latent = False,
                 **kwargs):
        labels = self.encode_text(prompt, self.text_embed)
        if noise_levels is None:
            noise_levels = (torch.pow(torch.arange(0, cur_noise_level, (cur_noise_level) / n_iter), exponent)).tolist()
            noise_levels.reverse()#(1 - torch.pow(torch.arange(cur_noise_level, 1, (1-cur_noise_level) / n_iter), exponent)).tolist()
        #noise_levels[0] = 0.99

        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]
        if latent is None:
            x_t = self.initialize_image(seeds, num_imgs, img_size, seed, img_channel)
        else:
            x_t = latent
        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        x0_pred_prev = None

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f
        if return_latent:
            return x0_pred
        else:
            x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
            return x0_pred_img

    def pred_image(self, noisy_image, labels, noise_level, class_guidance,c,controlnet,mask_token=None,not_control=None):
        num_imgs = noisy_image.size(0)
        noise_level = torch.from_numpy(noise_level)[:,None]
        noises = torch.cat([noise_level,noise_level],dim=0)
        class_guidance = torch.from_numpy(class_guidance)[:,None].to(self.device)
        #noises = torch.full((2 * num_imgs, 1), noise_level)
        if controlnet:
            x0_pred = self.model(torch.cat([noisy_image, noisy_image]),
                                 noises.to(self.device, self.model_dtype),
                                 labels.to(self.device, self.model_dtype),torch.cat([c,c]).to(self.device, self.model_dtype) if c is not None else None,mask_token=torch.cat([mask_token,mask_token]).to(self.device) if mask_token is not None else None,
                                 not_control=not_control)
        else:
            x0_pred = self.model(torch.cat([noisy_image, noisy_image]),
                                 noises.to(self.device, self.model_dtype),
                                 labels.to(self.device, self.model_dtype),mask_token=torch.cat([mask_token,mask_token]).to(self.device)if mask_token is not None else None)
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)
        return x0_pred

    def initialize_image(self, seeds, num_imgs, img_size, seed, img_channel):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(num_imgs, img_channel, img_size, img_size, dtype=self.model_dtype,
                               device=self.device, generator=generator)
        else:
            return seeds.to(self.device, self.model_dtype)
    def expand_scalar(self,scalar,shape):
        return scalar.reshape(-1,1,1,1).repeat(1,shape[1],shape[2],shape[3])

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        class_guidance = self.expand_scalar(class_guidance,x0_pred_label.shape)
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label
