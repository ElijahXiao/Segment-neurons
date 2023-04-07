import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from glob import glob
import json, os
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# custom dataset
class NeuronDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root # path of dataset
        self.transforms = transforms # data preprocessing
        self.regions = self.get_regions() # coordinates of all neurons         
        #self.imgs = np.array([imageio.imread(f) for f in glob(('train_data/neurofinder.00.00/images/*.tiff'))],dtype=np.float32)
        self.imgs = np.array([imageio.imread(f) for f in glob(os.path.join(root, "images/*.tiff"))],dtype=np.float32)
        # load all image files, sorting them to ensure that they are aligned
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        #self.img_dim = self.imgs.shape[1:]
        self.mask = self.get_mask()
        self.transforms = transforms
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def get_regions(self):
        #with open('train_data/neurofinder.00.00/regions/regions.json') as f:
        region_path = os.path.join(self.root, "regions/regions.json")
        with open(region_path) as f:
            regions = json.load(f)
        #print(regions)
        return regions

    def get_mask(self):
        def to_mask(coords):
            mask = np.zeros((512,512))
            for coord in coords:
                mask[coord[0]][coord[1]] = 1
            return mask
        
        #mask = np.array([to_mask(s['coordinates']) for s in self.regions],dtype=np.int64).sum(axis=0)
        mask = np.array([to_mask(s['coordinates']) for s in self.regions],dtype=np.int8).sum(axis=0)
        # deal with duplicate coordinates
        mask[mask > 1] = 1
        mask = torch.from_numpy(mask)
        return mask
        
    def __getitem__(self, idx):
        # transform
        #img = np.expand_dims(self.imgs[idx], axis = 0) # H * W -> C(1) * H * W
        #img = Image.fromarray(self.imgs[idx]) # PIL.image
        #img = img.convert("RGB")
        if self.transforms:
            pass
        
        #transform = transforms.ToTensor()
        #img = transform(img)
        #img = img[0]
        #img = img.unsqueeze(0)
        
        img = np.expand_dims(self.imgs[idx], axis = 0) # H * W -> C(1) * H * W
        img = torch.from_numpy(img)
        return img, self.mask
    
    def __len__(self):
        return len(self.imgs)

def build_loader(config, is_train, shuffle=True):
    if is_train: # split the training dataset into training and 
        dataset_train = NeuronDataset(config["train_data_path"])
    
        # Split training data into a training set and a validation set
        # the class cannot be put into split, wrong implementation  
        dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.2, random_state=42)
        
        # start from here, 
        data_loader_train = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=shuffle)
        # print(torch.cuda.memory_allocated()) 0
        data_loader_val = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False)
        return dataset_train, dataset_val, data_loader_train, data_loader_val
    else: # only return test data loader
        dataset_test = NeuronDataset(config["test_data_path"])
        data_loader_test = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)
        return dataset_test, data_loader_test
        
if __name__ == "__main__":
    config = {"batch_size":8,
              "num_epochs":10,
              "save_freq":1,
              "use_amp":True,
              "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
              "train_data_path":"data/train_data1/neurofinder.00.00",
              "test_data_path":"data/test_data1/neurofinder.00.00.test",
              "ckpt_path":"checkpoints/T2_batch8",
              "log_path":"logs/T2_batch8.txt"
            }
    _,_,_,_ = build_loader(config,True)

    
    
    
    