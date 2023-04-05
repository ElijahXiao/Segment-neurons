import matplotlib.pyplot as plt
import numpy as np
import math, time

import torch

import SwinUnet
from build import build_loader


# helper functions
def show_img(img, masks):
    plt.figure()
    plt.subplot(1, 2, 1)
    #plt.title("raw image")
    plt.title("Ground truth neuron positions") 
    plt.ylabel("y-coordinate")
    plt.xlabel("x-coordinate")
    
    #plt.imshow(imgs.sum(axis=0), cmap='gray')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted neuron positions")   
    plt.imshow(masks, cmap='gray')
    plt.suptitle("From raw data to segmented image",fontsize=15)
    plt.subplots_adjust(top=1.1)
    plt.show(block=True)
    
def show_4img(imgs):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    fig.suptitle("Predicted neuron positions at different epochs", fontsize=15)
    plt.subplots_adjust(top=0.9)
    
    ax[0,0].imshow(imgs[0], cmap='gray')
    ax[0,0].set_title("Epoch12")
    
    ax[0,1].imshow(imgs[1], cmap='gray')
    ax[0,1].set_title("Epoch24")
    
    ax[1,0].imshow(imgs[2], cmap='gray')
    ax[1,0].set_title("Epoch36")
    
    ax[1,1].imshow(imgs[3], cmap='gray')
    ax[1,1].set_title("Epoch48")
    
    
    plt.show()

def output_to_mask(img:torch.Tensor) -> np.ndarray: # for one image
    img = img.squeeze().permute(1, 2, 0).detach().numpy()
    max_indices = np.argmax(img, axis=-1)
    #output = max_indices.astype(np.int16)
    output = max_indices.astype(np.int8)
    
    return output

def outputs_to_masks(imgs: torch.Tensor) -> np.ndarray: # batch size * 2 * 512 * 512 -> batch size * 512 * 512
    imgs = imgs.permute(0, 2, 3, 1).detach().numpy()
    outputs = []
    for img in imgs:
        output = np.argmax(img, axis=-1)
        #output = max_indices.astype(np.int8)
        outputs.append(output)
    outputs = np.array(outputs, dtype=np.int8)
    return outputs  

def get_checkpoint():
    epochs = [5,20,25,30]
    outputs = []
    for epoch in epochs:
        checkpoint = torch.load(f"train_data/neurofinder.00.00/checkpoints/model_epoch_{epoch}.pt", map_location=torch.device('cpu'))
        output = checkpoint["output"].unsqueeze(0)
        output = output_to_mask(output)
        outputs.append(output)
    
    show_4img(outputs)
    
def load_checkpoint():
    pass
    
def save_checkpoint(epoch, model, optimizer, lr_scheduler,loss_scaler, save_path):
    save_state = {"model":model.state_dict(),
                  "optimizer":optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  }
    torch.save(save_state, save_path)
    #torch.save(save_state ,f"train_data/neurofinder.00.00/checkpoints/model_epoch_{epoch}.pt")
        
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

@torch.no_grad()
def test_forward(config):
    model = SwinUnet.SwinTransformerSys(img_size=512, in_chans=1, num_classes=2, window_size=8)
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config, is_train=True)
    
    sample, train_mask = dataset_train.__getitem__(0)
    print(sample.shape)
    sample = sample.view(1,1,512,512) # add batch size
    output = model(sample)    
    #print(output.shape) 1,2,512,512
    output_masks = outputs_to_masks(output)
    sample = sample.view(512,512)
    #imgs = [output_mask] * 4
    
    #show_4img(imgs)
    #show_img(sample, train_mask)
    show_img(sample, output_masks[0])
    

