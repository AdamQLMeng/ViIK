"""Global configuration"""

import os

import torch


DEFAULT_TORCH_DTYPE = torch.float32

DEFAULT_DATA_DIR = "./"
DATASET_DIR = os.path.join(DEFAULT_DATA_DIR, "data/")
