from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imread
from glob import glob
import imageio.v2 as imageio
import os
import torch
import model
import torchvision.transforms as transforms
import torch.nn as nn

def test(model, dataloader):
    model.eval