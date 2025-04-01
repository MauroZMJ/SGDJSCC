import torch
import numpy as np

def l2_normalization(signal):
    norm_2 = torch.linalg.norm(signal.reshape([signal.shape[0], -1]), ord=2, dim=1)
    per_element_power = norm_2 ** 2 / (signal.numel() / signal.shape[0])
    signal = signal / torch.sqrt(per_element_power[:, None, None, None])
    return signal
def through_fading_channel(signal,snr,channel_type='rayleigh'):
    #signal: Tensor
    #first conduct normalization on signal
    b,c,h,w = signal.shape
    signal = l2_normalization(signal)
    snr_scale = 10 ** (snr / 10)
    noise_scale = 1 / np.sqrt(snr_scale)

    if channel_type == 'rayleigh':
        fading_scalar_half = torch.abs((torch.randn_like(signal[:,:c//2,:]) + 1j * torch.randn_like(signal[:,:c//2,:]))/np.sqrt(2))
        fading_scalar = torch.cat([fading_scalar_half,fading_scalar_half],dim=1)
        fading_scalar_pow = fading_scalar ** 2
        Ws = fading_scalar_pow / (fading_scalar_pow+2*noise_scale**2)
        Wn = fading_scalar / (fading_scalar_pow+2*noise_scale**2)
        noise = torch.randn_like(signal)
        signal_scale = Ws/np.sqrt(1+noise_scale**2)
        noise_scale = noise_scale*Wn/np.sqrt(1+noise_scale**2)
        channel_output = signal*signal_scale + noise*noise_scale
        #conduct normalization on channel_output
        weighting_scalar = torch.sqrt(signal_scale**2+ noise_scale**2)
        channel_output = channel_output / weighting_scalar
        signal_scale = signal_scale / weighting_scalar
        noise_scale = noise_scale / weighting_scalar

        return channel_output,signal_scale,noise_scale
    elif channel_type == 'awgn':
        fading_scalar = 1
        fading_scalar_pow = fading_scalar ** 2
        Ws = fading_scalar_pow / (fading_scalar_pow+2*noise_scale**2)
        Wn = fading_scalar / (fading_scalar_pow+2*noise_scale**2)
        noise = torch.randn_like(signal)
        channel_output = signal*Ws/np.sqrt(1+noise_scale**2) + noise*noise_scale*Wn/np.sqrt(1+noise_scale**2)
        channel_output = l2_normalization(channel_output)
        return channel_output,fading_scalar_pow
    else:
        raise ValueError('channel type not supported')


