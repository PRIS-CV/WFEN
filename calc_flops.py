import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import models.arch.wfen as network

if __name__ == '__main__':
    
    model = network.WFEN()

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 128, 128)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input = input.to(device)
    macs, params = profile(model.to(device), inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("macs:", macs)
    print("params:", params)
