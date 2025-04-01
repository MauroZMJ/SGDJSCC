import os
import sys
import math
import torch
import numpy as np
from skimage import metrics
import lpips
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import torch.nn as nn
from torchmetrics.multimodal.clip_score import CLIPScore
from cleanfid import fid
from torchvision import utils as vutils
import shutil
class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * torch.log10(1 / mse)


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        batch_size = img1.shape[0]
        ssim_value = 0
        for b_idx in range(batch_size):
            ssim_value += ssim((img1[b_idx:b_idx+1,:]*255).to(torch.uint8).to(torch.float32),(img2[b_idx:b_idx+1,:]*255).to(torch.uint8).to(torch.float32))
            #ssim_value += metrics.structural_similarity(img1[b_idx,:].cpu().numpy().transpose((1,2,0)),img2[b_idx,:].cpu().numpy().transpose((1,2,0)),multichannel=True,data_range=1,channel_axis=2)
        return ssim_value/batch_size
class LPIPS:
    def __init__(self):
        self.name = "LPIPS"
        self.loss = lpips.LPIPS(net='alex').to(torch.device(0))
    def __call__(self,img1, img2):
        return torch.mean(self.loss(img1*2-1,img2*2-1))
class MSE_LPIPS(torch.nn.Module):
    def __init__(self):
        super(MSE_LPIPS, self).__init__()
        self.mse_loss = F.mse_loss
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.loss_weight = 0.1
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
    def forward(self, X, Y):
        return self.mse_loss(X,Y) + self.loss_weight*self.lpips_loss(2*X-1,2*Y-1)

class FID:
    def __init__(self, temp_1='/home/maojun/data/coco', temp_2='/home/maojun/data/coco'):
        self.name = "FID"
        self.temp_1 = temp_1
        self.temp_2 = temp_2
    def save_image_to_path(self,img_list,temp_path):
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        for i in range(len(img_list)):
            vutils.save_image(torch.tensor(img_list[i]), os.path.join(temp_path, '{}.png'.format(i)))
    def __call__(self, img1_list, img2_list,temp_1_name='temp1',temp_2_name='temp2'):
        temp_1_path = os.path.join(self.temp_1,temp_1_name)
        temp_2_path = os.path.join(self.temp_2,temp_2_name)

        self.save_image_to_path(img1_list,temp_1_path)
        self.save_image_to_path(img2_list,temp_2_path)
        score = fid.compute_fid(temp_1_path, temp_2_path)
        return score
class CLIP_score:
    def __init__(self):
        self.name = "CLIP_score"
        self.loss = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(torch.device(0))
    def __call__(self,img1, text1):
        return torch.mean(self.loss((torch.clamp(img1,0,1)*255).to(torch.int32),list(text1[0])))

class Classification_Accuracy:
    def __init__(self):
        self.name = "Classification Accuracy"

    def __call__(self, log_probs, label):
        y_pred = log_probs.data.max(1,keepdim=True)[1]
        correct_ratio = y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum() / len(label)
        return correct_ratio


class LPIPS_eav:
    def __init__(self):
        self.name = "LPIPS"
        self.loss = lpips.LPIPS(net='alex').to(torch.device(0))
    def __call__(self,img1, img2):
        return F.relu(1 - torch.mean(self.loss(img1*2-1,img2*2-1)))
class SSIM_eav:
    def __init__(self):
        self.name = "SSIM_eav"
        self.loss = ssim
    def __call__(self,img1, img2):
        return self.loss(img1,img2,data_range=1,size_average=True)
class MSE_eav:
    def __init__(self):
        self.name = "MSE_eav"
        self.loss = F.mse_loss
    def __call__(self,img1, img2):
        return F.relu(self.loss(img1*0,img2)-0.05)
class KLDIV_eav:
    def __init__(self):
        self.name = "KLDIV_eav"
        self.loss = nn.KLDivLoss(reduction='mean')
    def __call__(self,img1, img2):
        return F.relu(self.loss(img1,img2)-0.01)
class CE_eav:
    def __init__(self):
        self.name = "CE_eav"
        self.loss = nn.CrossEntropyLoss()
    def __call__(self,y_pred, y_true):
        return F.relu(self.loss(y_pred,y_true)-0.01)

