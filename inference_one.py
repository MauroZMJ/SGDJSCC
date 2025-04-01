from options import args_parser
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image as vsave_image
import matplotlib.pyplot as plt
import numpy as np
from utils.loss_function import  PSNR,SSIM,LPIPS,CLIP_score,FID
import torch.nn.functional as F
import os
from utils.plot import showOrigNoiseOut,save_image
from utils.utils import logger_setup,resize_preprocess_tensor,generate_mask,CropLongSide,cr_allocation
from torchvision import utils as vutils
from models.test_advanced_network.autoencoderkl import AutoencoderKL
from models.test_advanced_network.mask_diffusion_controlnet import MDTv2_ControlNet
from models.test_advanced_network.snr_prediction_net import Prediction_Model
from models.model_canny import Semantic_Communication_Model as Canny_Semantic_Communication_Model
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from utils.coco_dataset import Coco_Image_Canny_Uncertainty
from omegaconf import OmegaConf
import math
from utils.utils import split_image_v2,image_caption,generate_canny,merge_image_v2
class JSCC_model(nn.Module):
    def __init__(self,snr=10):
        super(JSCC_model, self).__init__()

        #define the JSCC model
        ddconfig = {'double_z': True, 'z_channels': 16, 'resolution': 128, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        self.vae = AutoencoderKL(ddconfig,16)

        #define the snr value
        self.snr = snr

        #define the snr prediction net
        self.snr_prediction_net = Prediction_Model()

        #load the JSCC model for edge map transmission
        self.canny_transmission_net = Canny_Semantic_Communication_Model(filters=[256, 2, 8,[128, 192, 256, 320],[2,2,6,2],[4,6,8,10]],snrdB=-5,channel='Dynamic_AWGN',channel_coding=False,modulating=False,in_feature=8192,size1=640,size2=320,model_type = 'vit_witt_adaln_radepth_4_uncertain',task_name='reconstruction')

        self.i = 0
    def normalize(self,x):
        batch_size,c,h,w = x.shape
        x = x.reshape(batch_size,-1)
        x = F.normalize(x,p=2,dim=1) * math.sqrt(x.shape[1])
        return x.reshape(batch_size,c,h,w)
    def encode(self,x):
        encode_features = self.vae.encode(x).latent_dist.mean
        #encode_features = self.normalize(encode_features)
        return encode_features
    def channel(self,encode_features):
        bsz, c, h, w = encode_features.shape
        norm_2 = torch.linalg.norm(encode_features.reshape([bsz, -1]), ord=2, dim=1)
        encode_features_hat = encode_features + torch.randn_like(encode_features) * torch.sqrt((norm_2**2/(encode_features.numel()/bsz))/(10**(self.snr/10))).reshape([-1,1,1,1]).repeat([1,c,h,w])
        return encode_features_hat
    def decode(self,encode_features_hat):
        #encode_features_hat = self.normalize(encode_features_hat)
        decode_image = self.vae.decode(encode_features_hat)[0]
        return decode_image
    def forward(self,x,gt_text,pipe,canny_data,canny_uncertainty,use_semantic,use_controlnet,use_text,use_gt_text,use_jscc_feature,use_gt_csi,controlnet_scale,mask_method,diffusion_step,step_style,cfg_method,guidance_scale,canny_cr,scaling_factor=15.45,):
        #process the canny images
        soft_edge_image = torch.mean(canny_data,axis=1,keepdim=True)
        soft_edge_uncertainty = torch.mean(canny_uncertainty,axis=1,keepdim=True)

        #encode the image
        latent_dist = self.vae.encode(x*2-1).latent_dist
        encode_features = self.normalize(latent_dist.mean/scaling_factor)
        encode_features_std = latent_dist.std

        #add noise at there
        snr_scale = 10 ** (self.snr / 10)
        signal_scale = snr_scale/(snr_scale+1)*torch.ones_like(encode_features[:,0:1,0,0])
        encode_features_hat = self.normalize(self.channel(encode_features))

        #denoising
        if use_semantic:
            if use_text:
                semantic_text = list(gt_text[0])
            else:
                semantic_text = ['' for _ in range(x.shape[0])]

            thresholded = soft_edge_image

            #do masking operation
            mask_token = generate_mask(encode_features,encode_features_std,thresholded,mask_method)
            encode_features_hat = encode_features_hat * mask_token.reshape([-1,1,mask_token.shape[1],mask_token.shape[2]]).repeat([1,encode_features_hat.shape[1],1,1]) if mask_token is not None else encode_features_hat


            power_scalar = torch.sqrt(torch.linalg.norm(encode_features_hat.reshape([x.shape[0],-1]), ord=2,axis=1) ** 2
                                      / (torch.sum(mask_token.reshape([-1,1,encode_features_hat.shape[2],encode_features_hat.shape[3]]).repeat([1,encode_features_hat.shape[1],1,1]).reshape(x.shape[0],-1),axis=1) if mask_token is not None else encode_features_hat[0,:].numel())
                                      ).reshape([-1,1,1,1]).repeat([1,encode_features_hat.shape[1],encode_features_hat.shape[2],encode_features_hat.shape[3]])

            #Step Matching
            if step_style == 'continuous':
                if use_jscc_feature:
                    if use_gt_csi:
                        cur_step = signal_scale.mean().item()
                        cur_step = 1 - cur_step
                        cur_snr = self.snr
                    else:
                        # for canny control net
                        predicted_signal_scale = (self.snr_prediction_net(encode_features_hat / power_scalar).reshape([-1, 1]) ** 2)
                        cur_step = 1 - predicted_signal_scale
                        cur_snr = 10*torch.log10((1/(cur_step)-1))
                else:
                    cur_step = 1
            elif step_style == 'discrete':
                if use_jscc_feature:
                    if use_gt_csi:
                        cur_step = torch.argmin(
                            torch.abs(pipe.scheduler.alphas_cumprod.unsqueeze(0).to(device) - signal_scale.reshape([-1, 1])),
                            axis=1).float().mean().int().item()
                    else:
                        #predicted_cur_step = torch.argmin(torch.abs(pipe.scheduler.alphas_cumprod.unsqueeze(0).to(device) - (self.snr_prediction_net(encode_features_hat / power_scalar).reshape([-1,1])**2)),axis=1).float().mean().int().item()
                        cur_step = torch.argmin(torch.abs(
                            pipe.scheduler.alphas_cumprod.unsqueeze(0).to(device) - (
                                        self.snr_prediction_net(encode_features_hat / power_scalar).reshape([-1, 1]) ** 2)),
                                                          axis=1).float().mean().int().item()
                else:
                    cur_step = 981

            if canny_cr != 'none':
                cr = torch.ones(x.size(0), 1).to(device) * round(float(config.transmission.canny_cr)*64)
                snr = torch.ones(x.size(0), 1).to(device) * cur_snr
                #the ground truth snr define here is only used for configuring the channel for edgemap transmission
                gt_snr = torch.ones(x.size(0), 1).to(device) * self.snr

                thresholded = (self.canny_transmission_net(torch.cat([thresholded,soft_edge_uncertainty],dim=1).to(device),gt_snr=gt_snr.to(device), snr=snr.to(device), cr=cr.to(device))).to(torch.float32)
                snr_threshold = (cur_snr<=-5).reshape(-1,1,1,1).repeat(1,1,thresholded.shape[2],thresholded.shape[3]).float()
                thresholded = (thresholded*(thresholded>config.th)*snr_threshold + thresholded*(1-snr_threshold)).float()

            #obtain the JSCC latent for edge map, this feature is fed into the diffusion denoiser
            canny_latent = self.vae.encode((thresholded * 2 - 1).repeat([1, 3, 1, 1]).to(device))[0].mean / scaling_factor

            if use_controlnet:
                #vutils.save_image(thresholded.repeat([1,3,1,1])[0,:],  './results/canny/%d.jpg'%(self.i)); self.i += 1
                image, denoised_latent = pipe.generate(prompt=semantic_text, num_imgs=1, class_guidance=guidance_scale, cfg_weighting_method=cfg_method,n_iter=40, not_control = [controlnet_scale for i in range(encode_features_hat.size(0))] + [0 for i in range(encode_features_hat.size(0))],
                                                scale_factor=1,latent=encode_features_hat / power_scalar if use_jscc_feature else torch.randn_like(encode_features_hat),negative_prompt = ['distorted, discontinuous, ugly, blurry, low resolution, deformed, bad quality, deformed' for i in range(len(semantic_text))],
                                                   return_latent=True, img_channel=16, img_size=16,alphas_cumprod=pipe.alphas_cumprod,curr_step=cur_step,c=canny_latent,controlnet=use_controlnet,mask_step=1,mask_token=mask_token,diffusion_step=diffusion_step,step_style=step_style)
            else:
                #the difference between use controlnet and not use controlnet is that the not use controlnet will not use the controlnet_scale.
                image, denoised_latent = pipe.generate(prompt=semantic_text, num_imgs=1, class_guidance=guidance_scale, cfg_weighting_method=config.diffusion.cfg_method,n_iter=40, not_control = [0 for i in range(encode_features_hat.size(0))] + [0 for i in range(encode_features_hat.size(0))],
                                                scale_factor=1,latent=encode_features_hat / power_scalar if use_jscc_feature else torch.randn_like(encode_features_hat),negative_prompt = ['distorted, discontinuous, ugly, blurry, low resolution, deformed, bad quality, deformed' for i in range(len(semantic_text))],
                                                   return_latent=True, img_channel=16, img_size=16,alphas_cumprod=pipe.alphas_cumprod,curr_step=cur_step,c=canny_latent,controlnet=use_controlnet,mask_step=1,mask_token=mask_token,diffusion_step=diffusion_step,step_style=step_style)
            #conduct the JSCC decoding process
            decode_image = (self.vae.decode(self.normalize(denoised_latent))[0]+1)/2
            caption_text = semantic_text
            return decode_image,caption_text

        caption_text = ['' for _ in range(x.shape[0])]
        decode_image = (self.vae.decode(self.normalize(encode_features_hat))[0]+1)/2
        return decode_image,caption_text
def test(data_loader,model,pipe,criterion_list,prefix='',fid_func=None,config=None):
    model.eval()
    loss_list = []
    logger.info("test started...")
    i = 1
    ori_img_list = []
    rec_img_list = []
    caption_list = []
    all_rec_img_list = []
    all_ori_img_list = []
    with torch.no_grad():
        #for H_bar_data,label_data,canny_data,canny_uncertainty in data_loader:
        for origin_img, label_data in data_loader:
            H_bar_data,meta = split_image_v2(origin_img)
            H_bar_data = H_bar_data.to(device)
            label_data = image_caption(caption_model, H_bar_data, device)
            print(label_data)
            canny_data, canny_uncertainty = generate_canny(H_bar_data,canny_net,device)
            # H_bar_data = H_bar_data.to(device)
            # label_data = image_caption()
            canny_data = canny_data.to(device)
            canny_uncertainty = canny_uncertainty.to(device)
            # forward
            model_output,caption_text = model(H_bar_data,pipe=pipe,gt_text=label_data,canny_data=canny_data,canny_uncertainty=canny_uncertainty,
                                              use_semantic=config.model.condition_setting.use_semantic,use_controlnet=config.model.condition_setting.use_controlnet,use_text=config.model.condition_setting.use_text,use_gt_text=config.model.condition_setting.use_gt_text,canny_cr=config.transmission.canny_cr,
                                              use_jscc_feature=config.model.condition_setting.use_jscc_feature,use_gt_csi=config.transmission.use_gt_csi,controlnet_scale=config.model.diffusion.controlnet_scale,mask_method=config.transmission.mask_method,diffusion_step=config.model.diffusion.diffusion_step,step_style=config.model.diffusion.step_style,cfg_method=config.model.diffusion.cfg_method,guidance_scale=config.model.diffusion.guidance_scale,)

            reconstructed = merge_image_v2(model_output, meta)
            loss_instance = [criterion(H_bar_data, model_output).item() if not isinstance(criterion,CLIP_score) else criterion(model_output, label_data).item() for criterion in criterion_list]
            loss_list.append(loss_instance)
            ori_img_list.append(H_bar_data.detach().cpu().numpy())
            rec_img_list.append(model_output.detach().cpu().numpy())
            caption_list.extend(caption_text)
            if not os.path.exists(base_root + 'image/%s' % (prefix)):
                os.mkdir(base_root + 'image/%s' % (prefix))
            vutils.save_image(reconstructed, base_root + 'image/%s/kodim'%(prefix) + str(i).zfill(2)+'.png')
            if i >= config.dataset_config.test_num:
                break
            i = i + 1


    fid_score = 0
    model.train()
    return np.concatenate([np.mean(loss_list,axis=0),[fid_score]],axis=0)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    device = torch.device(0)
    set_seed(2025)
    args = args_parser()
    config = OmegaConf.load(args.config_root)

    base_root = './results/'
    version_string = '_v11'
    task_save_root = 'Eva_Ori_%s_step_%s_use_semantic_%d_use_control_%d_use_text_%d_use_gt_text_%d_use_csi_%d_diffusion_step_%d_mask_%s_canny_%s_jsccf_%d_sam_%d_cfg_%.1f_%s_ctrl_scale_%.2f_%s' % (
    config.dataset, config.model.diffusion.step_style, config.model.condition_setting.use_semantic, config.model.condition_setting.use_controlnet, config.model.condition_setting.use_text, config.model.condition_setting.use_gt_text,
    config.transmission.use_gt_csi, config.model.diffusion.diffusion_step,
    config.transmission.mask_method, config.transmission.canny_cr, config.model.condition_setting.use_jscc_feature, config.dataset_config.use_sam_canny, config.model.diffusion.guidance_scale, config.model.diffusion.cfg_method,
    config.model.diffusion.controlnet_scale,version_string)
    base_root = base_root + task_save_root + '/'
    # Create logger folder
    if not os.path.exists(base_root):
        os.mkdir(base_root)
    sub_file_name = ['log/', 'model/', 'image/', 'tensorboard/']
    for sub_file in sub_file_name:
        if not os.path.exists(base_root + sub_file):
            os.mkdir(base_root + sub_file)
    # Create logger
    logger = logger_setup(base_root + 'log/%s.log' % (task_save_root))

    # process the file root
    dataset_root = config.dataset_config.dataset_root
    fid_root = config.dataset_config.fid_root
    model_root = config.model.model_root
    import random;
    import string
    folder_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    fid_root = os.path.join(fid_root, folder_name)
    if not os.path.exists(fid_root):
        os.mkdir(fid_root)

    if config.dataset == 'coco_blip':
        trans_linnaeus = transforms.Compose([CropLongSide(), transforms.Resize((128, 128)), transforms.ToTensor()])
        dataset_test = Coco_Image_Canny_Uncertainty(dataset_root+'test_2017/val2017/images',
                                       dataset_root+'test_2017/val2017/sam_softedge' if config.dataset_config.use_sam_canny else dataset_root+'test_2017/val2017/full_softedge',
                                    dataset_root + 'test_2017/val2017/softedge_Uncertainty',
                                       dataset_root+'test_2017/val2017/blip_captions_val2017.json',
                                       transform=trans_linnaeus)
        dataset_valid = Coco_Image_Canny_Uncertainty(dataset_root+'test_2017/test2017/images',
                                        dataset_root+'test_2017/test2017/sam_softedge' if config.dataset_config.use_sam_canny else dataset_root+'test_2017/test2017/full_softedge',
                                        dataset_root + 'test_2017/test2017/softedge_Uncertainty',
                                        dataset_root+'test_2017/test2017/blip_captions_test2017_new.json',
                                        transform=trans_linnaeus)
    elif config.dataset == 'kodak':
        #Here we conduct center crop on the image dataset, or we can also transmit patch by patch
        trans_linnaeus = transforms.Compose([CropLongSide(), transforms.Resize((128, 128)), transforms.ToTensor()])
        dataset_train = datasets.ImageFolder('./assets', transform=trans_linnaeus)
        dataset_test = datasets.ImageFolder('./assets', transform=trans_linnaeus)
        train_dataset_length = len(dataset_train)
        train_size, validate_size = int(0.9 * train_dataset_length), int(train_dataset_length - int(0.9 * train_dataset_length))
        dataset_train, dataset_valid = torch.utils.data.random_split(dataset_train, [train_size, validate_size])

    valid_dataloader = torch.utils.data.DataLoader(dataset_valid, batch_size=config.dataset_config.batch_size, shuffle=False, num_workers=config.dataset_config.num_workers)
    test_dataloader  = torch.utils.data.DataLoader(dataset_test, batch_size=config.dataset_config.batch_size, shuffle=False, num_workers=config.dataset_config.num_workers)


    model = JSCC_model()
    model.to(device)
    pretrained_model_state_dict = torch.load(os.path.join(model_root,'JSCC_model.pth'))
    model.load_state_dict(pretrained_model_state_dict)

    if config.model.condition_setting.use_semantic:
        #load the caption model
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco",
                                                                      torch_dtype=torch.float16)
        caption_model.processor = processor
        caption_model.to(device)
        #load the canny extractor
        from models.test_advanced_network.muge_model import Mymodel as muge_model
        canny_net = muge_model()
        canny_net.load_state_dict(torch.load(os.path.join(model_root,'muge-epoch-19-checkpoint.pth'))['state_dict'])
        canny_net.to(device)

        #load the foundation diffusion model
        from models.test_advanced_network.mask_diffusion import MDTv2
        denoiser = MDTv2(depth=12, hidden_size=512, patch_size=1, num_heads=8)
        denoiser.load_state_dict(torch.load(os.path.join(model_root, 'diffusion_backbone.pth'))['model_ema'])
        #using controlnet
        if config.model.condition_setting.use_controlnet:
            denoiser = MDTv2_ControlNet(base_model=denoiser, copy_blocks_num=6,hidden_size=512)
            denoiser.load_state_dict(torch.load(os.path.join(model_root,'diffusion_controlnet.pth'))['model_ema'])
        denoiser.to(device)

        #load the clip model
        import clip
        clip_model, preprocess = clip.load("ViT-L/14")
        clip_model = clip_model.to(device)

        #load the JSCC model
        ddconfig = {'double_z': True, 'z_channels': 16, 'resolution': 128, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        vae = AutoencoderKL(ddconfig, 16)
        vae.load_state_dict(model.vae.state_dict(), False)
        vae.to(device)

        #define the pipeline for diffusion denoising
        from models.test_advanced_network.diffusion_element_wise import DiffusionGenerator
        sem_pipeline = DiffusionGenerator(denoiser, vae, clip_model,device, torch.float32)

        canny_transmission_net = None

    else:
        caption_model = None
        sem_pipeline = None
        compression_net = None
        canny_net = None
        canny_transmission_net = None
        sam_text_model = None

    SNR_list = [-5,-15,-10,-5,0,5,10,15]
    performance = []
    hyper_parameter = {-15:[0.05, 0.35],
                       -10:[0.10, 0.25],
                       -5 :[0.15, 0.25],
                       0  :[0.20, 0.25],
                       5  :[0.20, 0.25],
                       10 :[0.20, 0.25],
                       15 :[0.20, 0.25]}
    forward_fn_list = {'d2d':test}
    criterion_name_list = {'LPIPS': LPIPS(), 'PSNR': PSNR(), 'SSIM': SSIM(),'CLIP':CLIP_score()}
    version_string = '_v5'
    for SNR in SNR_list:
        model.snr = SNR
        logger.info('Current SNR: %d dB'%SNR)

        snr_performance = []
        for forward_fn in forward_fn_list.keys():
            logger.info('LPIPS , PSNR, SSIM testing...')
            config.model.diffusion.controlnet_scale, config.th = hyper_parameter[SNR]
            lpips_performance = test(data_loader=test_dataloader, model=model,
                                     pipe=sem_pipeline,
                                     criterion_list=[criterion_name_list[cri_name] for cri_name in criterion_name_list.keys()],
                                     prefix='snr_%d'%(SNR),
                                     fid_func=FID(temp_1=fid_root,temp_2=fid_root),
                                     config=config)
            for i,cri_name in enumerate(criterion_name_list.keys()):
                logger.info('%s performance: %.4f'%(cri_name,lpips_performance[i]))
            logger.info('FID performance: %.4f'%lpips_performance[4])
            snr_performance.append([lpips_performance[i] for i in range(len(lpips_performance))])
        performance.append(snr_performance)
    def make_table(performance,name_list,index=0):
        from prettytable import PrettyTable
        table_head = ['Forward Fn/SNR']
        table_head.extend(SNR_list)
        table = PrettyTable(table_head)
        i = 0
        for forward_fn in name_list:
            item_list = [forward_fn]
            item_list.extend(performance[:,i,index])
            table.add_row(item_list)
            i += 1
        return table


    import numpy as np
    performance = np.round(np.array(performance), decimals=2)
    logger.info('LPIPS performance')
    logger.info(make_table(performance,forward_fn_list.keys(),index=0))

    logger.info('PSNR performance')
    logger.info(make_table(performance,forward_fn_list.keys(),index=1))
    logger.info('SSIM performance')
    logger.info(make_table(performance,forward_fn_list.keys(),index=2))
    logger.info('CLIP performance')
    logger.info(make_table(performance,forward_fn_list.keys(),index=3))
    logger.info('FID performance')
    logger.info(make_table(performance,forward_fn_list.keys(),index=4))


    logger.info(performance)
    #remove fid root
    import shutil
    shutil.rmtree(fid_root)



