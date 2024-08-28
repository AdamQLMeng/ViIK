#! /bin/bash
# Create virtual environment
conda create -n flows python=3.9
conda activate flows
# Install dependencies
pip install torch==2.0.1 torchvision==0.15.2
pip install pandas pytorch_lightning matplotlib tqdm scikit-learn scipy tensorboardX torchdiffeq ninja imageio einops jrl==0.0.9 qpth -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install FrEIA==0.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install PyQT5 -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt-get install libxcb-xinerama0
conda install theano
