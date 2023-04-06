import os, gc, time

import torch
import torch.nn as nn

import SwinUnet
from build import build_loader
from utils import save_checkpoint, timeSince, test_forward
from evaluate import validate, eval_metrics

from timm.utils import AverageMeter

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, loss_scaler, start_time):
    model.train()
    loss_meter = AverageMeter()
    
    for idx, (imgs, labels) in enumerate(data_loader):
        
        imgs = imgs.to(config["device"])
        labels = labels.to(config["device"])
                
        #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_amp"]): # run the forward pass under autocast
        with torch.cuda.amp.autocast(enabled=config["use_amp"]):
            outputs = model(imgs)
        loss = criterion(outputs, labels)        
    
        loss_meter.update(loss.item(), config["batch_size"])
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()
        
    lr_scheduler.step(loss_meter.avg)
    print(f"{timeSince(start_time, epoch / config['num_epochs'])} ({epoch} {epoch / config['num_epochs'] * 100}) {loss_meter.avg}")
        
       
 
def main(config):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert config["device"] == torch.device('cuda')
    
    model = SwinUnet.SwinTransformerSys(img_size=512, in_chans=1, num_classes=2, window_size=8)
    model.to(config["device"]) # model memory 122340864 bytes
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config, is_train=True)    
    print(torch.cuda.memory_allocated())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    
    print("start training")
    start_time =time.time()
    for epoch in range(1, config["num_epochs"]+1):
        # AttributeError: 'RandomSampler' object has no attribute 'set_epoch'
        #data_loader_train.sampler.set_epoch(epoch) 
        print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
        #train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, loss_scaler, start_time)
        if epoch % config["save_freq"] == 0:
            #save_checkpoint(epoch, model, optimizer, lr_scheduler,loss_scaler,config["ckpt_path"])
            pass
        
        precision_avg, recall_avg, f1_avg, iou_avg, loss_avg = validate(config, data_loader_val, model)
        print("val: ",precision_avg, recall_avg, f1_avg, iou_avg, loss_avg)
        
    torch.cuda.synchronize()
        
        

if __name__ == '__main__':
    config = {"batch_size":8,
              "num_epochs":5,
              "save_freq":1,
              "use_amp":True,
              "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
              "train_data_path":"data/train_data1/neurofinder.00.00",
              "test_data_path":"data/test_data1/neurofinder.00.00.test",
              "ckpt_path":"checkpoints/T2_batch8",
              "log_path":"logs/T2_batch8.txt"
            }
    # generate a checkpoint folder 
    os.makedirs(config["ckpt_path"], exist_ok=True)
    os.makedirs(config["log_path"], exist_ok=True)
    
    #test_forward(config)
    main(config)