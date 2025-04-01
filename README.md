# SGDJSCC
## Description
This repository contains the code for the paper "Semantics Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission". 
## Installation
To quickly run the code, please run ```install.sh``` line by line. 
We use Python 3.9 and CUDA 11.8 to run the code. All GPUs with more than 12GB memory should be ok. 
# 🔮 Inference
Before run the code, you should first download the pretrained model from <a href="https://huggingface.co/murjun/SGDJSCC/tree/main">here</a> and put it in the ```checkpoint``` folder. 

We currently provide the code for AWGN case (with SNR information or without SNR information). 
Then you can run the following command to transmit an image:
```bash
python inference_one.py --config_root configs/inference.yaml
```

```inference_config.py``` is for running with preprocessed batch data. We will soon update the preprocess script for further evaluation. 

Please note that this code has been thoroughly organized. If you find anything that seems to be wrong, please raise an issue or send me an email.
# 📝 TODO List
- [x] Release environment setup, inference code, and model checkpoints.
- [ ] Release the preprocessing script.
- [ ] Training guideline to fine-tune the diffusion model or controlnet for wireless image transmission.

# 📚 BibTeX
For any reproduction or use of the code, please cite the following paper:
```
@article{zhang2025semantics,
  title={Semantics-Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission},
  author={Zhang, Maojun and Wu, Haotian and Zhu, Guangxu and Jin, Richeng and Chen, Xiaoming and G{\"u}nd{\"u}z, Deniz},
  journal={arXiv preprint arXiv:2501.01138},
  year={2025}
}
```

# 🙏 Acknowledgements
The development of SGDJSCC has been greatly inspired by the following amazing works and teams:
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)