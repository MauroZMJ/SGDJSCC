import torch.nn as nn
import torch
import torch.nn.functional as F

from .channel import RayleighChannel,AWGNChannel,IdealChannel,MisoFadingChannel,MisoFadingChannel_Realvalue,Dynamic_AWGN,Dynamic_Fading
import numpy as np
import math

from models.test_advanced_network.autoencoderkl import AutoencoderKL

class Semantic_Communication_Model(nn.Module):
    def __init__(self,filters=[32,64,128],snrdB=25,channel='Rayleigh',channel_coding=True,modulating=False,image_size=128,in_feature=3200,size1=1280,size2=640,model_type='attention',task_name='reconstruction',num_classes=10,flatten_size=128):
        super(Semantic_Communication_Model, self).__init__()
        self.model_type = model_type
        self.semantic_encoder = Semantic_Encoder(filters=filters,model_type=model_type,task_name=task_name,image_size=image_size)
        self.channel_encoder = Channel_Encoder(in_features=in_feature, size1=size1, size2=size2,channel_coding=channel_coding)
        self.modulator = QAMModulator(order=256,modulating=modulating)
        if channel == 'Rayleigh':
            self.channel = RayleighChannel(snrdB=snrdB)
        elif channel == 'AWGN':
            self.channel = AWGNChannel(snrdB=snrdB)
        elif channel == 'Ideal':
            self.channel = IdealChannel(snrdB=snrdB)
        elif channel == 'MisoFading':
            self.channel = MisoFadingChannel(snrdB = snrdB)
        elif channel == 'MisoFadingReal':
            self.channel = MisoFadingChannel_Realvalue(snrdB = snrdB)
        elif channel == 'Dynamic_AWGN':
            self.channel = Dynamic_AWGN(snrdB=snrdB)
        elif channel == 'Dynamic_Fading':
            self.channel = Dynamic_Fading(snrdB=snrdB)
        self.demodulator = QAMDemodulator(order=256,modulating=modulating)
        self.channel_decoder = Channel_Decoder(in_features=size2, size1=in_feature, size2=size1,channel_coding=channel_coding)
        #filters.reverse()
        self.semantic_decoder = Semantic_Decoder(filters=filters,model_type=model_type,task_name=task_name,num_classes=num_classes,flatten_size=flatten_size,image_size=image_size)
        if 'image_feature' in model_type:
            ddconfig = {'double_z': True, 'z_channels': 16, 'resolution': 128, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                        'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

            self.vae = AutoencoderKL(ddconfig, 16)
            self.vae.load_state_dict(torch.load('/home/zhangmaojun/Desktop/Semantics_Guided_Video_Transmission/results/model/final.ckpt')['state_dict'],False)
            for name, param in self.vae.named_parameters():
                param.requires_grad = False

    def transmitter(self,inputs,snr,cr):
        if 'adaln' in self.model_type:
            x_feature, snr_embedding,mask = self.semantic_encoder(inputs,snr,cr)
        else:
            x_feature = self.semantic_encoder(inputs,snr,cr)
            mask = None
        feature_shape = x_feature.shape
        x_flatten_feature = torch.reshape(x_feature,[feature_shape[0],-1])
        x_code = self.channel_encoder(x_flatten_feature,mask)
        x_code_modulated = self.modulator(x_code)
        if 'adaln' in self.model_type:
            return x_code_modulated, feature_shape,snr_embedding,mask
        else:
            return x_code_modulated, feature_shape,mask
    def receiver(self,inputs,feature_shape,snr,cr,snr_embedding=None,mask=None,image_noisy_feature=None):
        y_demodulated = self.demodulator(inputs)
        x_decode = self.channel_decoder(y_demodulated,mask)
        x_decode_feature = torch.reshape(x_decode,feature_shape)
        x_decode_feature = x_decode_feature * mask if 'ra' in self.model_type else x_decode_feature
        if 'adaln' in self.model_type:
            x_restore = self.semantic_decoder(x_decode_feature,snr,cr,snr_embedding,image_noisy_feature)
        else:
            x_restore = self.semantic_decoder(x_decode_feature,snr,cr,snr_embedding,image_noisy_feature)
        return x_restore
    def forward(self,x,return_eav_result=False,gt_snr=None,snr=None,cr=None,image_noisy_feature=None,images=None):
        #x = x.repeat([1,3,1,1])
        gt_snr_db = gt_snr
        snr_db = snr
        snr_scale = 10**(snr/10)
        if 'adaln' in self.model_type:
            x_code_modulated,feature_shape,snr_embedding,mask = self.transmitter(x,snr_scale,cr)
        else:
            x_code_modulated,feature_shape,mask = self.transmitter(x,snr_scale,cr)
        y = self.channel(x_code_modulated,gt_snr_db)
        if 'image_feature' in self.model_type:
            with torch.no_grad():
                image_feature = self.vae.encode(images*2-1).latent_dist.mean/15.45
                image_feature_shape = image_feature.shape
                image_feature = torch.reshape(image_feature,[image_feature_shape[0],-1])
                image_noisy_feature = self.channel_decoder(self.channel(image_feature,snr_db)).reshape(image_feature_shape)
        x_restore = self.receiver(y, feature_shape, snr_scale, cr, snr_embedding if 'adaln' in self.model_type else None,mask,image_noisy_feature)
        return x_restore
class Semantic_Encoder(nn.Module):
    def __init__(self,filters=[[3,256,9,4,4],[256,256,5,2,2],[256,256,5,2,2],[256,256,5,2,2],[256,64,5,2,2]],model_type = 'cnn',task_name='reconstruction',image_size=128):
        super(Semantic_Encoder, self).__init__()
        self.encoder = []
        if task_name == 'reconstruction':
            if 'vit' in model_type:
                N = filters[0]
                patch_size = filters[1]
                window_size = filters[2]
                embed_dims = filters[3]
                depths = filters[4]
                num_heads = filters[5]
                #from models.test_advanced_network.witt_encoder import create_encoder
                if 'adaln' in model_type:
                    from models.test_advanced_network.revised_witt.witt_encoder import create_encoder
                    self.encoder = create_encoder(image_dims=image_size, C=N, model_type=model_type,
                                                  patch_size=patch_size, window_size=window_size, embed_dims=embed_dims,
                                                  uncertain='uncertain' in model_type,
                                                  depths=depths, num_heads=num_heads)
        else:
            assert "task name definition error"
        self.encoder = nn.Sequential(*self.encoder) if  'vit' not in model_type else self.encoder
        self.model_type = model_type
    def forward(self,x,snr=None,cr=None):
        if 'vit' in self.model_type:
            return self.encoder(x,snr,cr)
        else:
            return self.encoder(x)
class Normalize_layer(nn.Module):
    def __init__(self):
        super(Normalize_layer, self).__init__()
    def forward(self,x,mask=None):
        if mask is not None:
            return F.normalize(x,p=2,dim=1)*torch.sqrt(torch.sum(mask,dim=[1,2])).reshape([-1,1]).repeat(1,x.shape[1])
        else:
            return F.normalize(x,p=2,dim=1) * math.sqrt(x.shape[1])
class Channel_Encoder(nn.Module):
    def __init__(self, in_features, size1, size2,channel_coding=True):
        super(Channel_Encoder, self).__init__()
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.channel_coding = channel_coding
        self.normalize_layer = Normalize_layer()
    def forward(self, x, mask=None):
        if self.channel_coding:
            x = self.linear1(x)
            x = F.relu(x)
            output = self.linear2(x)
            return self.normalize_layer(output)
        else:
            return self.normalize_layer(x,mask)
class QAMModulator(nn.Module):
    def __init__(self, order=256,modulating=True):
        super(QAMModulator, self).__init__()
        self.order = order
        self.n_bit = int(math.log2(self.order))
        self.avg_power = self.cal_avg_power(self.order)
        self.modulating = modulating
    def cal_avg_power(self,order):
        avg_power = []
        for i in range(int(math.sqrt(order)/2)):
            avg_power.append((i+0.5)**2)
        avg_power = np.array(avg_power)
        avg_power = np.sqrt(2*np.sum(avg_power)/len(avg_power))
        return avg_power

    def forward(self,x):
        '''
        :param x: is a float vector varies from -1 to 1
        :return: the modulated vector
        '''
        #doing quantization for the input
        if self.modulating:
            inputs = ((x+1)/2*255).int()
            avg_power = self.avg_power
            # bit split
            lower_half_bit_mask = 2 ** (self.n_bit//2) - 1
            upper_half_bit_mask = lower_half_bit_mask << (self.n_bit//2)

            lower_bit = torch.bitwise_and(inputs, lower_half_bit_mask)
            upper_bit = (torch.bitwise_and(inputs, upper_half_bit_mask))>>( self.n_bit//2)
            output = torch.stack([lower_bit,upper_bit],axis=2)
            output = torch.bitwise_xor(output, output>>1)
            # center to zero and power normalization
            output = (output.float() - (self.order ** 0.5)/2 + 0.5) / avg_power
            return output
        else:
            return x
class QAMDemodulator(nn.Module):
    def __init__(self, order=256,modulating=True):
        super(QAMDemodulator, self).__init__()
        self.order = order
        self.n_bit = int(math.log2(self.order))
        self.avg_power = self.cal_avg_power(self.order)
        self.modulating = modulating
    def cal_avg_power(self,order):
        avg_power = []
        for i in range(int(math.sqrt(order)/2)):
            avg_power.append((i+0.5)**2)
        avg_power = np.array(avg_power)
        avg_power = np.sqrt(2*np.sum(avg_power)/len(avg_power))
        return avg_power
    def forward(self,inputs):
        if self.modulating:
            avg_power = self.avg_power
            yhat = inputs*avg_power + (self.order ** 0.5)/2 - 0.5
            #QAM detection
            yhat_decode = torch.floor(yhat+0.5).int()
            min_val = 0
            max_val = self.order ** 0.5 -1
            yhat_grey = F.hardtanh(yhat, min_val=min_val, max_val=max_val).int()

            # undo zero-centering
            nhat_grey = yhat_grey.clone()
            for i in range(self.n_bit//2-1):
                nhat_grey >>= 1
                yhat_grey = torch.bitwise_xor(yhat_grey, nhat_grey)
            demodulated_result = torch.bitwise_or(yhat_grey[:,:,1]<<(self.n_bit//2),yhat_grey[:,:,0])
            demodulated_output = demodulated_result / 255 * 2 - 1
            return demodulated_output
        else:
            return inputs
class Channel_Decoder(nn.Module):
    def __init__(self, in_features, size1, size2,channel_coding=True):
        super(Channel_Decoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        self.channel_coding = channel_coding
        self.normalize_layer = Normalize_layer()
    def forward(self, x,mask=None):
        if self.channel_coding:
            x1 = self.linear1(x)
            x2 = F.relu(x1)
            x3 = self.linear2(x2)
            x4 = F.relu(x3)
            x5 = self.linear3(x4)
            output = self.normalize_layer(x1 + x5)
            return output
        else:
            return self.normalize_layer(x,mask)
class Semantic_Decoder(nn.Module):
    def __init__(self,filters=[[64,32,3,2,0],[64,64,3,1,0],[32,64,3,1,0],[3,32,2,1,0]],model_type = 'cnn',task_name='reconstruction',flatten_size=8192,num_classes=10,image_size=128):
        super(Semantic_Decoder, self).__init__()
        self.decoder = []
        if task_name == 'reconstruction':
            if 'vit' in model_type:
                patch_size = filters[1]
                N = filters[0]
                window_size = filters[2]
                embed_dims = filters[3];
                depths = filters[4];
                num_heads = filters[5];
                if 'adaln' in model_type:
                    from models.test_advanced_network.revised_witt.witt_decoder import create_decoder
                    self.decoder = create_decoder(image_dims=image_size, C=N, model_type=model_type,
                                                  patch_size=patch_size, window_size=window_size, embed_dims=embed_dims,
                                                  depths=depths, num_heads=num_heads)
                #self.decoder = create_decoder(image_dims=image_size,C=N,model_type=model_type,patch_size=patch_size,window_size=window_size,embed_dims=embed_dims,depths=depths,num_heads=num_heads)
            if 'vit' not in model_type:
                self.decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder) if 'vit' not in model_type else self.decoder
        self.model_type = model_type
    def forward(self,x,snr=None,cr=None,snr_embedding=None,image_noisy_feature=None):
        if  'vit' in self.model_type:
            if 'adaln' in self.model_type:
                return self.decoder(x,snr,cr,snr_embedding,image_noisy_feature=image_noisy_feature)
            else:
                return self.decoder(x,snr,cr,snr_embedding,image_noisy_feature=image_noisy_feature)
        else:
            return self.decoder(x)
