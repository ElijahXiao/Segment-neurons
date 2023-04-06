# Load Data
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import imageio.v2 as imageio
import os
import torch
import SwinUnet
import torchvision.transforms as transforms
import torch.nn as nn
import gc, time, math
from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
#exit()

# helper functions
def show_img(img, masks):
    print("fuck img")
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
    plt.show(block=False)
    
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

def output_to_mask(img:torch.Tensor) -> np.ndarray: # batch size * 2 * 512 * 512 -> 512 * 512
    img = img.squeeze().permute(1, 2, 0).detach().numpy()
    # squeeze remove any dimension of 1
    max_indices = np.argmax(img, axis=-1)
    #output = max_indices.astype(np.int16)
    output = max_indices.astype(np.int8)
    
    return output

def outputs_to_mask(imgs: torch.Tensor) -> np.ndarray: # batch size * 2 * 512 * 512 -> 512 * 512
    imgs = imgs.cpu().permute(0, 2, 3, 1).detach().numpy()
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
        #output = checkpoint["output"].unsqueeze(0)
        output = checkpoint["output"] # batch_size * 2 * 512 * 512
        output = outputs_to_mask(output)
        outputs.append(output)
    
    show_4img(outputs)
        
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
    
"""
********************************
# data 
********************************
"""

"""
class NeuronDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, img_dim=512, datatype="train", transforms=None):
        self.root = root # path of dataset
        self.transforms = transforms # data preprocessing
        self.regions = self.get_regions() # coordinates of all neurons
        #if datatype != "test":
        #    self.data_folder = f"{img_paths}/train_data/neurofinder.00.00.test/images"
        #else:
        #    self.data_folder = f"{img_paths}/test_data/neurofinder.00.00/images"
         
        #self.imgs = np.array([imageio.imread(f) for f in sorted(glob('images/*.tiff'))],dtype=np.float32)
        self.imgs = np.array([imageio.imread(f) for f in glob(('train_data/neurofinder.00.00/images/*.tiff'))],dtype=np.float32)
        # load all image files, sorting them to ensure that they are aligned
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.img_dim = self.imgs.shape[1:]
        self.mask = self.get_mask()
        self.transforms = transforms
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def print_mask(self):
         print(len(self.mask))
        
    def get_regions(self):
        #with open('/userhome/30/dfxiao/FYP/train_data/neurofinder.00.00/regions/regions.json') as f:
        with open('train_data/neurofinder.00.00/regions/regions.json') as f:
        #with open('regions/regions.json') as f:
        
            regions = json.load(f)
        #print(regions)
        return regions

    def get_mask(self):
        def to_mask(coords):
            mask = np.zeros(self.img_dim)
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
        img = np.expand_dims(self.imgs[idx], axis = 0) # H * W -> C(1) * H * W
        # transform
        img = torch.from_numpy(img).to(self.device)
        #self.mask
        if self.transforms:
            pass
        return img, self.mask
    
    def __len__(self):
        return len(self.imgs)
"""

class NeuronDataset(torch.utils.data.Dataset):
    def __init__(self, root="data/train_data1/neurofinder.00.00", transforms=None):
        self.root = root # path of dataset
        self.transforms = transforms # data preprocessing
        self.regions = self.get_regions() # coordinates of all neurons         
        self.imgs = np.array([imageio.imread(f) for f in glob(('data/train_data1/neurofinder.00.00/images/*.tiff'))],dtype=np.float32)
        #self.imgs = np.array([imageio.imread(f) for f in glob(os.path.join(root, "images/*.tiff"))],dtype=np.float32)
        # load all image files, sorting them to ensure that they are aligned
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        #self.img_dim = self.imgs.shape[1:]
        self.mask = self.get_mask()
        self.transforms = transforms
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
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
        
    #def __getitem__(self, idx): 
    #    img = np.expand_dims(self.imgs[idx], axis = 0) # H * W -> C(1) * H * W
    #    # transform
    #    img = torch.from_numpy(img).to(self.device)
    #    #self.mask
    #    if self.transforms:
    #        pass
    #    return img, self.mask.to(self.device)
        
    def __getitem__(self, idx):
        #img_path = os.path.join(self.root, f"images/image{'{:05d}'.format(idx)}.tiff")
        # transform
        #print(os.path.join(self.root, "images/*.tiff'"))
        img = np.expand_dims(self.imgs[idx], axis = 0) # H * W -> C(1) * H * W
        img = torch.from_numpy(img)
        if self.transforms:
            pass
        
        return img, self.mask
    
    def __len__(self):
        return len(self.imgs)
           
train_data = NeuronDataset()     
#print(type(sample)) ndarray
#print(sample.dtype) float64
#print(train_mask.dtype) float64
#train_mask = torch.from_numpy(train_mask)

#show_img(sample, train_mask)

"""
****************************************************************
# model
****************************************************************
"""
model = SwinUnet.SwinTransformerSys(img_size=512, in_chans=1, num_classes=2, window_size=8)
model.to(device)
print(torch.cuda.memory_allocated())
#device = next(model.parameters()).device
#print(device)
#summary(model, input_size=(1,512,512)) # get model summary
#sample = sample.astype(float) 
#sample = torch.from_numpy(sample)
#sample = sample.view(1,1,512,512)
#print(sample.shape) # 1 1 512 512

#img = model.forward(sample.float())
#for name, param in model.named_parameters():
#    print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

#print(img.shape) # B N H W , 1 2 512 512

# test forward
def test_forward():
    sample, train_mask = train_data.__getitem__(0)
    sample = sample.view(1,1,512,512) # add batch size
    output = model(sample)    
    #print(output.shape) 1,2,512,512
    output_mask = outputs_to_mask(output)
    #print(output_mask)
    sample = sample.view(512,512)
    imgs = [output_mask] * 4
    
    show_4img(imgs)
    #show_img(sample, output_mask)

def test_forward_new(model, train_data): # unfinished
    sample, train_mask = train_data.__getitem__(0)    
    
    sample = sample.view(512,512)
    sample = sample.numpy()
    print(sample.shape)
    show_img(sample, sample)
    
    sample = sample.view(1,1,512,512) # add batch size
    output = model(sample)    
    #print(output.shape) 1,2,512,512
    output_mask = outputs_to_mask(output)
    sample = sample.view(512,512)
    #imgs = [output_mask[0]] * 4
    #sample = sample.cpu().numpy()
    #train_mask = train_mask.cpu().numpy()
    print(sample.shape)
    #show_img(sample, sample)
    
    #show_4img(imgs)
    sample = sample.cpu().numpy()
    train_mask = train_mask.numpy()
    show_img(train_mask, train_mask)
    train_mask = torch.from_numpy(train_mask).to(device)
    train_mask = train_mask.cpu().numpy()
    show_img(train_mask, train_mask)

    #show_img(sample, train_mask) 
    #show_img(sample, sample)
    
    
#test_forward_new(model, train_data)


"""
********************************
# training
********************************
"""
#criterion = nn.NLLLoss2d()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#log_softmax = nn.LogSoftmax(dim=1)
#img = log_softmax(img)
#train_mask=torch.tensor(train_mask, dtype=torch.long) 
#train_mask = train_mask.unsqueeze(0)
#print(img.shape) 1,2,512,512
#print(train_mask.shape) 1,512,512
#print(img.dtype) float32             
#print(train_mask.dtype) #int64
#loss = criterion(img, train_mask)
#print(loss)
print(torch.cuda.memory_allocated())
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
print(torch.cuda.memory_allocated())
#print(len(train_dataloader.dataset)) 3024
#print(len(train_dataloader.sampler)) 3024
#test_dataloader = DataLoader(test_data, batch_size=63, shuffle=True)

#print(len(train_dataloader)) 48
#for data in train_dataloader:
#    print(type(data)) list
#    print(len(data)) 2, imgs labels
#    print(len(data[0])) batch_size = 64
#    break
    


def training(device,num_epochs=30, save_res=[5, 15,20,25,30], print_every = 5): # num_epochs = total samples / batch_size
    # some initialization before training
    use_amp=True
    loss_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print_loss_total = 0  # Reset every print_every
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    
    print("start training")
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    start =time.time()
    for epoch in range(1, num_epochs+1):
        model.train()
        optimizer.zero_grad()
        train_loss = 0
        for b_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp): # run the forward pass under autocast
                
                outputs = model(imgs)
              
            loss = criterion(outputs+(1e-8), labels)
                
            loss_scaler.scale(loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_dataloader)
        lr_scheduler.step(avg_train_loss)
        print_loss_total += train_loss
        print(avg_train_loss)
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"{timeSince(start, epoch / num_epochs)} ({epoch} {epoch / num_epochs * 100}) {print_loss_avg}")
            #print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
            
        if epoch in save_res:
            sample, _ = train_data.__getitem__(0)
            sample = sample.view(1,1,512,512) # add batch size
            output = model(sample.detach())
            save_state = {"model":model.state_dict(),
                          "optimizer":optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'scaler': loss_scaler.state_dict(),
                          'epoch': epoch,
                          "output":output
                          }
            #torch.save(save_state ,f"train_data/neurofinder.00.00/checkpoints/model_epoch_{epoch}.pt")
    torch.cuda.synchronize()
        
            # Validation
            #model.eval()
            #val_loss = 0
            #with torch.no_grad():
            #for i, (x, y) in enumerate(val_loader):
            #pred = model(x)
            #loss = loss_fn(pred, y)
            #val_loss += loss.item()
            #avg_val_loss = val_loss / len(val_loader)
            #
            #scheduler.step(avg_val_loss) # Adjust learning rate if necessary
            
            #Print the training status
            #print('Epoch: {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, avg_train_loss, avg_val_loss))
            #
            ## 8. Evaluation
            #test_dataset = ImageDataset(train=False, transform=...)
            #test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            #model.eval()
            #total_correct, total_samples = 0, 0
            #with torch.no_grad():
            #for x, y in test_loader:
            #pred = model(x)
            #_, predicted = torch.max(pred.data, 1)
            #total_samples += y.size(0)
            #total_correct += (predicted == y).sum().item()
            #
            ## Print prediction accuracy, precision, and recall
            #precision = total_correct / total_samples
            #recall = total_correct / len(test_dataset)
            #f_score = 2 * precision * recall / (precision + recall)
            #print('Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F-score: {:.2f}%'.format(
            #100 * total_correct / total_samples, 100 * precision, 100 * recall, 100 * f_score))
        
training(device)
#get_checkpoint()

# testing
