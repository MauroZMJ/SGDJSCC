from pytorch_msssim import ssim,ms_ssim,SSIM,MS_SSIM
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import lpips
def ssim_loss(y_true,y_pred):
    ssim_val = ssim( y_true, y_pred, data_range=1, size_average=True,nonnegative_ssim=True)
    return 1 - ssim_val
def ms_ssim_loss(y_true,y_pred):
    ms_ssim_loss = ms_ssim(y_true, y_pred, data_range=1, size_average=True)
    return 1 - ms_ssim_loss
def ssim_mse_loss(y_true,y_pred):
    ssim_loss = 1 - ssim(y_true, y_pred, data_range=1, size_average=True)
    mse_loss = F.mse_loss(y_true,y_pred)
    return 0.5*ssim_loss + 0.5*mse_loss
def mse_loss(y_true,y_pred):
    mse_loss = F.mse_loss(y_true,y_pred)
    return mse_loss
def eav_mse_loss(y_true,y_pred):
    return -F.mse_loss(y_true,y_pred)
def rate_distortion_loss(y_true,y_pred):
    lambda_val = 0.1
    N, _, H, W= y_true.size()
    out = {}
    num_pixels = N * H * W
    out["bpp_loss"] = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in y_pred["likelihoods"].values()
    )
    out["mse_loss"] = F.mse_loss(y_pred["x_hat"], y_true)
    out["loss"] = lambda_val * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
    return out
class secure_loss_reconstruction(nn.Module):
    def __init__(self,loss_fn_destination,loss_fn_eavesdropper,weight=0.3):
        super(secure_loss_reconstruction, self).__init__()
        self.loss_fn_destination = loss_fn_destination
        self.loss_fn_eavesdropper = loss_fn_eavesdropper
        self.weight = weight
    def forward(self,y_true,y_pred):
        destination_loss = self.loss_fn_destination (y_true,y_pred[:,:,:,:,0])
        eavesdropper_loss = self.loss_fn_eavesdropper(y_true, y_pred[:, :, :, :, 1])
        return destination_loss + self.weight * eavesdropper_loss
class secure_loss_classification(nn.Module):
    def __init__(self,loss_fn_destination,loss_fn_eavesdropper,weight=5,num_classes=10):
        super(secure_loss_classification, self).__init__()
        self.loss_fn_destination = loss_fn_destination
        self.loss_fn_eavesdropper = loss_fn_eavesdropper
        self.weight = weight
        self.reference_vector = None
        self.batch_size = 64
        self.num_classes = num_classes
    def forward(self,y_pred,y_true):
        if self.reference_vector is None or self.batch_size !=y_pred.shape[0]:
            self.reference_vector = 1 / self.num_classes * torch.ones_like(y_pred[:,:,0]).to(torch.device(0))
            self.batch_size = y_pred.shape[0]
        destination_loss = self.loss_fn_destination (y_pred[:,:, 0],y_true)
        eavesdropper_loss = self.loss_fn_eavesdropper(F.log_softmax(y_pred[:,:,1],dim=1),self.reference_vector)
        return destination_loss + self.weight * eavesdropper_loss
        #return destination_loss + self.weight*F.relu(eavesdropper_loss-0.1)
class LPIPS_loss(nn.Module):
    def __init__(self):
        super(LPIPS_loss, self).__init__()
        self.loss_fn = lpips.LPIPS(net='alex').to(torch.device(0))
    def forward(self,img1, img2):
        return torch.mean(self.loss(img1*2-1,img2*2-1))
# def secure_mse(y_true,y_pred):
#     destination_mse_loss = F.mse_loss(y_true,y_pred[:,:,:,:,0])
#     eavesdropper_mse_loss = F.mse_loss(y_true*0,y_pred[:,:,:,:,1])
#     return destination_mse_loss + 0.3*eavesdropper_mse_loss