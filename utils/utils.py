import torch
from torch import nn, optim

import logging
from torchvision import transforms
from torch_fidelity.metric_fid import fid_features_to_statistics,fid_statistics_to_metric
import numpy as np
import torch.nn.functional as F
import re
class CropLongSide:
    def __call__(self, img):
        size = min(img.size)
        return transforms.CenterCrop((size,size))(img)
def logger_setup(log_root):
    # 创建一个logger
    logger = logging.getLogger('mytest')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_root)
    fh.setLevel(logging.DEBUG)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('semantic communication')
    return logger
def configure_optimizers(net,lr):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=lr,
    )
    return optimizer, aux_optimizer

def preprocess_tensor(img_batch,preprocess):
    batch_size = img_batch.shape[0]
    preprocess_img_batch = []
    for i in range(batch_size):
        preprocess_img_batch.append(preprocess(transforms.ToPILImage()(img_batch[i])))
    return torch.stack(preprocess_img_batch,dim=0)
def resize_preprocess_tensor(img_batch,preprocess):
    batch_size = img_batch.shape[0]
    preprocess_img_batch = []
    resize_transform = transforms.Resize((256,256))
    compose_transform = transforms.Compose([resize_transform,transforms.ToPILImage(),preprocess])
    for i in range(batch_size):
        preprocess_img_batch.append(compose_transform(img_batch[i]))
    return torch.stack(preprocess_img_batch,dim=0)
def cr_allocation(inputs,cr_budget,minimum_cr=0.05,allocation_scheme='pixel'):
    pixel_num = torch.mean(inputs.reshape([inputs.shape[0],-1]),dim=1)
    if allocation_scheme == 'pixel':
        cr = pixel_num / torch.sum(pixel_num) * pixel_num.shape[0] * (cr_budget-minimum_cr) + minimum_cr
    #import torch functional
    return cr
def cal_fid_mse_func(latent_step_total,ori_latent_total):
    step_num = latent_step_total.shape[1]
    data_num = latent_step_total.shape[0]
    fid_list = []
    latent_ori = ori_latent_total.reshape(data_num, -1)
    state_ori = fid_features_to_statistics(latent_ori)
    state_step_list = []
    mse_list = []
    for step_idx in range(step_num):
        print(step_idx)
        latent_step = latent_step_total[:,step_idx].reshape(data_num,-1)
        state_step = fid_features_to_statistics(latent_step)
        state_step_list.append(state_step)
        mse_list.append(F.mse_loss(latent_step,latent_ori))
        fid = fid_statistics_to_metric(state_step, state_ori)
        fid_list.append(fid)
    state_ori_list = [state_ori for i in range(step_num)]
    #fid_list = fid_statistics_to_metric_batch(state_step_list, state_ori_list)
    return fid_list,mse_list

def fid_statistics_to_metric(stat_1, stat_2):
    mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
    mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
    assert mu1.ndim == 1 and mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.ndim == 2 and sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    diff = mu1 - mu2
    tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid
def fid_statistics_to_metric_batch(stat_1, stat_2):
    la = 1
    mu1,sigma1 = torch.tensor(np.stack([state['mu'] for state in stat_1],axis=0)), torch.tensor(np.stack([state['sigma'] for state in stat_1],axis=0))
    mu2,sigma2 = torch.tensor(np.stack([state['mu'] for state in stat_2],axis=0)), torch.tensor(np.stack([state['sigma'] for state in stat_2],axis=0))
    diff = mu1 - mu2
    tr_covmean = torch.sum(torch.real(torch.sqrt(torch.linalg.eigvals(sigma1@sigma2))),axis=1)
    fid = torch.sum(diff**2,axis=1) + torch.sum(torch.diagonal(sigma1,dim1=1,dim2=2),axis=1) + torch.sum(torch.diagonal(sigma2,dim1=1,dim2=2),axis=1) - 2 * tr_covmean
    return fid

def keep_top_n_elements(tensor, n):
    flattened_tensor = tensor.view(-1)
    _, indices = torch.topk(flattened_tensor, n)
    mask_tensor = torch.zeros_like(flattened_tensor)
    mask_tensor[indices] = 1
    mask_tensor = mask_tensor.view(tensor.shape)
    return mask_tensor
def extract_floats_from_string(input_string):
    # 使用正则表达式匹配浮点数
    float_pattern = r"[-+]?\d*\.\d+|\d+"
    floats_list = re.findall(float_pattern, input_string)
    # 将字符串转换为浮点数并返回
    floats = [float(num) for num in floats_list]
    return floats
def generate_mask(x_latent,x_latent_std,canny,mask_method='none'):
    if mask_method == 'none':
        #mask_token = torch.ones_like(x_latent[:,0,:,:])
        return None
    elif 'canny' in mask_method:
        canny_score = F.avg_pool2d(canny[:,0,:],[8,8])
        mask_ratio = extract_floats_from_string(mask_method)[0]
        mask_token = keep_top_n_elements(canny_score,int(mask_ratio*canny_score.numel()))
        return mask_token.to(torch.float32).to(x_latent.device)
    elif 'random' in mask_method:
        mask_ratio = extract_floats_from_string(mask_method)[0]
        mask_token = torch.rand_like(x_latent[:,0,:,:]) < mask_ratio
        return mask_token.to(torch.float32).to(x_latent.device)
    elif 'latent' in mask_method:
        latent_score = torch.norm(x_latent,dim=1)
        mask_ratio = extract_floats_from_string(mask_method)[0]
        mask_token = keep_top_n_elements(latent_score,int(mask_ratio*latent_score.numel()))
        return mask_token.to(torch.float32).to(x_latent.device)
    elif 'std' in mask_method:
        latent_score = -torch.sum(x_latent_std,dim=1)
        mask_ratio = extract_floats_from_string(mask_method)[0]
        mask_token = keep_top_n_elements(latent_score,int(mask_ratio*latent_score.numel()))
        return mask_token.to(torch.float32).to(x_latent.device)

    else:
        raise ValueError('mask method not supported')


def split_image_v2(img):
    """
    输入：1x3xHxW 的图像张量
    输出：分块张量（Nx3x128x128）和位置信息列表
    """
    assert img.dim() == 4 and img.size(0) == 1 and img.size(1) == 3, "输入应为 [1, 3, H, W] 形状"

    _, C, H, W = img.shape
    patches = []
    positions = []

    # 计算分块步长
    stride_h = max(128 - (H % 128), 1) if H > 128 else H
    stride_w = max(128 - (W % 128), 1) if W > 128 else W

    # 生成分块坐标
    for i in range((H - 1) // stride_h + 1):
        for j in range((W - 1) // stride_w + 1):
            # 计算实际起始位置
            h_start = max(0, i * stride_h)
            w_start = max(0, j * stride_w)

            # 动态调整终点
            h_end = min(h_start + 128, H)
            w_end = min(w_start + 128, W)

            # 处理不足128的情况：向前扩展
            if h_end - h_start < 128:
                h_start = max(0, h_end - 128)
            if w_end - w_start < 128:
                w_start = max(0, w_end - 128)

            # 提取分块
            patch = img[:, :, h_start:h_end, w_start:w_end]
            patches.append(patch)
            positions.append((h_start, w_start))

    return torch.cat(patches, dim=0), (H, W, positions)
def merge_image_v2(patches, meta):
    H_orig, W_orig, positions = meta
    device = patches.device
    merged = torch.zeros(1, 3, H_orig, W_orig, device=device)
    count = torch.zeros(1, 3, H_orig, W_orig, device=device)

    for idx, (h_start, w_start) in enumerate(positions):
        # 添加通道维度处理
        patch = patches[idx].unsqueeze(0)  # 转换为 [1, 3, 128, 128]

        h_end = min(h_start + 128, H_orig)
        w_end = min(w_start + 128, W_orig)

        # 计算分块中的有效区域
        patch_h = h_end - h_start
        patch_w = w_end - w_start

        # 正确维度索引
        merged[:, :, h_start:h_end, w_start:w_end] += patch[:, :, :patch_h, :patch_w]
        count[:, :, h_start:h_end, w_start:w_end] += 1

    return merged / count.clamp(min=1)
def image_caption(model,image_tensor,device):
    processor = model.processor
    input_images = processor([transforms.ToPILImage()(image_tensor[i,:]) for i in range(len(image_tensor))],return_tensors="pt")
    input_images['pixel_values'] = [input_images['pixel_values'][i].to(device, torch.float16) for i in range(len(input_images['pixel_values']))]
    input_images['pixel_values'] = torch.stack(input_images['pixel_values'],dim=0)
    generated_ids = model.generate(**input_images, max_new_tokens=20)
    text_list = processor.batch_decode(generated_ids, skip_special_tokens=True)
    text_list = [prompt.strip() for prompt in text_list]
    return [text_list]
def generate_canny(raw_images,canny_net,device):
    images = F.interpolate(raw_images, size=(512, 512), mode='bilinear', align_corners=True).to(device)
    canny_data = F.interpolate(canny_net.one_shot_inference(images),size=(128,128))
    canny_bias_list = []
    bsz = raw_images.shape[0]
    for j in range(11):
        #print(j)
        bias = j / 10.0 * torch.ones(bsz).to(device)
        image_with_bias = F.interpolate(canny_net.one_shot_inference_with_bias(images, bias=bias),
                                        size=(128, 128)).cpu()
        canny_bias_list.append(image_with_bias)
    canny_list = torch.stack(canny_bias_list, dim=0).mean(dim=0)
    return canny_data,canny_list


