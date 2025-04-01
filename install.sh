conda create -n ptest python=3.9
conda activate ptest
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.44.2
pip install einops
pip install matplotlib
pip install numpy==1.23.2
pip install scikit-image
pip install lpips
pip install pytorch-msssim
pip install torchmetrics
pip install clean-fid
pip install torch-fidelity
pip install pytorch_lightning==2.4.0
pip install diffusers==0.26.3
pip install huggingface-hub==0.25.1
pip install timm==0.9.16
pip3 install xformers==0.0.22.post7
pip install omegaconf
pip install efficientnet_pytorch
pip install openai-clip