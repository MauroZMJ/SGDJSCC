import argparse
import numpy as np
def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epoch', type=int, default=300, help="rounds of training")
    # parser.add_argument('--batch_size', type=int, default=32, help="rounds of training")
    # parser.add_argument('--num_workers', type=int, default=4, help="number of workers for the data loading process")
    # parser.add_argument('--gpu', type=int, default=0, help="gpu index selection")
    # parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    # parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    # parser.add_argument('--snrdB', type=float, default=10, help="learning rate")
    # parser.add_argument('--weight', type=float, default=0.3, help="secure loss weight")
    # parser.add_argument('--channel', type=str, default='Ideal', help="channel type")
    # parser.add_argument('--snr', type=float, default=10, help="snr")
    # parser.add_argument('--channel_coding', action='store_true', help="whether channel coding or not")
    # parser.add_argument('--size', type=int, default=640, help="gpu index selection")
    # parser.add_argument('--vae_idx', type=int, default=0, help="gpu index selection")
    #
    # parser.add_argument('--modulating', action='store_true', help="whether modulating or not")
    # parser.add_argument('--progress_bar', action='store_true', help="whether show the progress bar or not")
    # parser.add_argument('--loss_func', type=str, default='psnr', help="loss function")
    # parser.add_argument('--task_name', type=str, default='reconstruction', help="specific task name")
    # parser.add_argument('--secure_transmission', action='store_true', help="whether resist the eavesdropper")
    # parser.add_argument('--debug', action='store_true', help="debug mode")
    # parser.add_argument('--use_semantic', action='store_true', help="debug mode")
    # parser.add_argument('--use_gt_text', action='store_true', help="debug mode")
    # parser.add_argument('--use_lr_guide', action='store_true', help="debug mode")
    # parser.add_argument('--use_text', action='store_true', help="debug mode")
    # parser.add_argument('--lr_guide_factor', type=float, default=100, help="secure loss weight")
    # parser.add_argument('--diffusion_step', type=int, default=50, help="gpu index selection")
    # parser.add_argument('--test_num', type=int, default=100, help="gpu index selection")
    # parser.add_argument('--model_type', type=str, default='proposed', help="specific task name")
    # parser.add_argument('--use_controlnet', action='store_true', help="debug mode")
    # parser.add_argument('--mask_method', type=str, default='none', help="specific task name")
    # parser.add_argument('--canny_cr', type=str, default='none', help="specific task name")
    # parser.add_argument('--use_jscc_feature', action='store_true', help="debug mode")
    # parser.add_argument('--use_sam_canny', action='store_true', help="debug mode")
    # parser.add_argument('--use_gt_csi', action='store_true', help="debug mode")
    #
    # parser.add_argument('--server_name', type=str, default='matchlab-1', help="specific task name")
    #
    # parser.add_argument('--step_style', type=str, default='continuous', help="specific task name")
    #
    # parser.add_argument('--model_path', type=str, default='continuous', help="specific task name")
    #
    # parser.add_argument('--controlnet_scale', type=float, default=1, help="secure loss weight")
    # parser.add_argument('--guidance_scale', type=float, default=1, help="secure loss weight")
    # parser.add_argument('--cfg_method', type=str, default='constant', help="specific task name")
    #
    # parser.add_argument('--fading_method', type=str, default='propose', help="specific task name")
    #
    # parser.add_argument('--channel_type', type=str, default='Gaussian', help="specific task name")
    # parser.add_argument('--precoding_method', type=str, default='ZF', help="specific task name")
    # parser.add_argument('--Nt', type=int, default=16, help="gpu index selection")
    # parser.add_argument('--M', type=int, default=8, help="gpu index selection")
    parser.add_argument('--config_root', type=str, default='configs/inference.yaml', help="gpu index selection")



    args = parser.parse_args()
    return args
