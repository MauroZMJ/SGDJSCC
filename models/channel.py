import torch
import random
import torch.nn as nn
import math
device = torch.device(0)
class RayleighChannel(nn.Module):
    def __init__(self, snrdB=None,eav_ratio_dB=15):
        super().__init__()
        self.noise_var =1 /( 10 ** (snrdB / 10))
        self.eav_ratio = 10**(eav_ratio_dB/10)
    def forward(self,Tx_sig):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(self.noise_var), size=Tx_sig.shape).to(device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig
    def eavesdropper(self,Tx_sig):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(self.noise_var*self.eav_ratio), size=Tx_sig.shape).to(device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

class AWGNChannel(nn.Module):
    def __init__(self, snrdB=None,eav_ratio_dB=15):
        super().__init__()
        self.noise_var = 1 /( 10 ** (snrdB / 10))
        self.eav_ratio = 10**(eav_ratio_dB/10)
    def forward(self,Tx_sig):
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(self.noise_var), size=Tx_sig.shape).to(device)
        return Rx_sig
    def eavesdropper(self,Tx_sig):
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(self.noise_var*self.eav_ratio), size=Tx_sig.shape).to(device)
        return Rx_sig

class Dynamic_AWGN(nn.Module):
    def __init__(self, snrdB=None):
        super().__init__()
        self.noise_var = 1 /( 10 ** (snrdB / 10))
    def forward(self,Tx_sig,snr):
        noise_var = 1 /( 10 ** (snr / 10))
        Rx_sig = Tx_sig + torch.randn_like(Tx_sig)*torch.sqrt(noise_var.repeat([1,Tx_sig.shape[1]]))
        return Rx_sig

class Dynamic_Fading(nn.Module):
    def __init__(self, snrdB=None):
        super().__init__()
        self.noise_var = 1 /( 10 ** (snrdB / 10))
    def forward(self,Tx_sig,snr):
        noise_var = 1 / (10 ** (snr / 10))
        bsz,m = Tx_sig.shape
        fading_scalar = torch.abs(torch.view_as_complex((torch.randn_like(Tx_sig)/math.sqrt(2)).reshape([bsz,m//2,2]))).repeat(1,1,2).reshape(bsz,m)
        noise = torch.randn_like(Tx_sig)
        noise_scale = torch.sqrt(noise_var.repeat([1,Tx_sig.shape[1]]))
        Rx_sig = Tx_sig*fading_scalar + noise*noise_scale
        return Rx_sig
class MisoFadingChannel(nn.Module):
    def __init__(self, snrdB=None,Nt=8):
        super().__init__()
        self.noise_var =1 /( 10 ** (snrdB / 10))
        self.Nt = Nt
    def generate_channel_instance(self):
        H_real = torch.normal(0, math.sqrt(1 / 2 /self.Nt), size=[1,self.Nt]).to(device)
        H_imag = torch.normal(0, math.sqrt(1 / 2 /self.Nt), size=[1,self.Nt]).to(device)
        H = torch.view_as_complex(torch.cat([H_real,H_imag],axis=2))
        U,S,V = torch.svd(H)
        return H,U,S,V
    def transmit_precoding(self,H,V):
        return torch.matmul(H,V)[0,0]
    def add_gussion_noise(self,Tx):
        noise_real = torch.normal(0, math.sqrt(self.noise_var/2), size=Tx.shape).to(device).unsqueeze(dim=-1)
        noise_imag = torch.normal(0, math.sqrt(self.noise_var/2), size=Tx.shape).to(device).unsqueeze(dim=-1)
        noise = torch.view_as_complex(torch.cat([noise_real,noise_imag],axis=-1))
        return noise
    def receive_recovering(self,H,y):
        la = 1
    def forward(self,Tx_sig):
        shape = Tx_sig.shape
        Tx_complex = torch.view_as_complex(Tx_sig.view(shape[0], -1, 2))
        H,U,S,V = self.generate_channel_instance()
        HV = self.transmit_precoding(H,V)
        HVx = HV*Tx_complex
        HVx_n = HVx + self.add_gussion_noise(HVx)
        Rx_sig = torch.view_as_real(torch.matmul(HVx_n.unsqueeze(dim=-1), torch.inverse(U*S))).view(shape)
        return Rx_sig
    def eavesdropper(self,Tx_sig):
        shape = Tx_sig.shape
        Tx_complex = torch.view_as_complex(Tx_sig.view(shape[0], -1, 2))
        H_obj,U_obj,S_obj,V_obj = self.generate_channel_instance()
        H_eav,U_eav,S_eav,V_eav = self.generate_channel_instance()
        HV = self.transmit_precoding(H_eav, V_obj)
        HVx = HV * Tx_complex
        HVx_n = HVx + self.add_gussion_noise(HVx)
        Rx_sig = torch.view_as_real(torch.matmul(HVx_n.unsqueeze(dim=-1), torch.inverse(U_eav * S_eav))).view(shape)
        return Rx_sig
class MisoFadingChannel_Realvalue(nn.Module):
    def __init__(self, snrdB=None,Nt=8):
        super().__init__()
        self.noise_var =1 /( 10 ** (snrdB / 10))
        self.Nt = Nt
    def generate_channel_instance(self,batch_size=1):
        H = torch.normal(0, math.sqrt(1  /self.Nt), size=[batch_size,1,self.Nt]).to(device)
        U,S,V = torch.svd(H)
        return H,U,S,V
    def transmit_precoding(self,H,V):
        return torch.matmul(H,V)
    def add_gussion_noise(self,Tx):
        noise = torch.normal(0, math.sqrt(self.noise_var), size=Tx.shape).to(device)
        return noise
    def receive_recovering(self,H,y):
        la = 1
    def forward(self,Tx_sig):
        shape = Tx_sig.shape
        #Tx_complex = torch.view_as_complex(Tx_sig.view(shape[0], -1, 2))
        H,U,S,V = self.generate_channel_instance(batch_size = shape[0])
        HV = self.transmit_precoding(H,V)
        HVx = HV[:,:,0].repeat([1,shape[1]]) * Tx_sig
        HVx_n = HVx + self.add_gussion_noise(HVx)
        Rx_sig = torch.matmul(HVx_n.unsqueeze(dim=-1), torch.inverse(U*S.unsqueeze(dim=-1))).view(shape)
        #Rx_sig = torch.view_as_real(torch.matmul(HVx_n.unsqueeze(dim=-1), torch.inverse(U*S))).view(shape)
        return Rx_sig
    def eavesdropper(self,Tx_sig):
        shape = Tx_sig.shape
        H_obj,U_obj,S_obj,V_obj = self.generate_channel_instance(batch_size = shape[0])
        H_eav,U_eav,S_eav,V_eav = self.generate_channel_instance(batch_size = shape[0])
        HV = self.transmit_precoding(H_eav, V_obj)
        HVx = HV[:,:,0].repeat([1,shape[1]]) * Tx_sig
        HVx_n = HVx + self.add_gussion_noise(HVx)
        Rx_sig = torch.matmul(HVx_n.unsqueeze(dim=-1), torch.inverse(U_eav * S_eav.unsqueeze(dim=-1))).view(shape)
        return Rx_sig


class RicianChannel(nn.Module):
    def __init__(self, snrdB=None):
        super().__init__()
        self.noise_var =1 /( 10 ** (snrdB / 10))
    def forward(self,Tx_sig):
        shape = Tx_sig.shape
        mean = math.sqrt(self.K / (self.K + 1))
        std = math.sqrt(1 / (self.K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(self.noise_var), size=Tx_sig.shape).to(device)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig


class IdealChannel(nn.Module):
    def __init__(self, snrdB=None,K=1):
        super().__init__()
        self.snr = 10 ** (snrdB / 10)
        self.K = 1
        self.eav_ratio = 10**(15/10)
    def forward(self,Tx_sig):
        #power normalization
        return Tx_sig
    def eavesdropper(self,Tx_sig):
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(self.eav_ratio), size=Tx_sig.shape).to(device)
        return Rx_sig

