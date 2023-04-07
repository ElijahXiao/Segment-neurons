import os, gc, time

import torch
import torch.nn as nn

from timm.utils import AverageMeter

import SwinUnet
from build import build_loader
from utils import save_checkpoint, timeSince, test_forward
from evaluate import validate
from logger import create_logger


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, loss_scaler):
    model.train()
    loss_meter = AverageMeter()
    start_time = time.time()
    for idx, (imgs, labels) in enumerate(data_loader):
        
        imgs = imgs.to(config["device"])
        labels = labels.to(config["device"])
                
        #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_amp"]): # run the forward pass under autocast
        #with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config["use_amp"]):
        outputs = model(imgs)
        #has_nan = torch.isnan(outputs).any() False
        #has_inf = torch.isinf(outputs).any() False
        #has_only_ones_and_zeros = torch.all(torch.logical_or(labels == 0, labels == 1)) True
        
        loss = criterion(outputs, labels)        
        
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        loss_meter.update(loss.item(), config["batch_size"])

        optimizer.zero_grad()
        
    lr_scheduler.step(loss_meter.avg)
    
    lr = optimizer.param_groups[0]['lr']
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f'Train: [{epoch}/{config["num_epochs"]}]\t'
        f'lr {lr:.6f}\t'
        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'epoch time {time.time() - start_time}'
        f'mem {memory_used:.0f}MB')      
       
 
def main(config):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    assert config["device"] == torch.device('cuda')
    
    model = SwinUnet.SwinTransformerSys(img_size=512, in_chans=1, num_classes=2, window_size=8)
    #logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    
    model.to(config["device"]) # model memory 122340864 bytes
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config, is_train=True)    
   
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    
    logger.info("Start training")
    start_time =time.time()
    for epoch in range(1, config["num_epochs"]+1):
        # AttributeError: 'RandomSampler' object has no attribute 'set_epoch'
        #data_loader_train.sampler.set_epoch(epoch) 
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, lr_scheduler, loss_scaler)
        if epoch % config["save_freq"] == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler,loss_scaler,config["ckpt_path"])
            
        precision_avg, recall_avg, f1_avg, iou_avg, loss_avg = validate(config, data_loader_val, model, logger)
        logger.info(f"{timeSince(start_time, epoch / config['num_epochs'])} (progress:{epoch / config['num_epochs'] * 100})")
         
    torch.cuda.synchronize()
        
        

if __name__ == '__main__':
    config = {"module_name": "main",
              "batch_size":8,
              "num_epochs":1,
              "save_freq":1,
              "use_amp":True,
              "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
              "train_data_path":"data/train_data1/neurofinder.00.00",
              "test_data_path":"data/test_data1/neurofinder.00.00.test",
              "ckpt_path":"checkpoints/T1_batch8",
              "log_name":"T1_batch8.txt"
            }
    # generate a checkpoint folder 
    os.makedirs(config["ckpt_path"], exist_ok=True)
    logger = create_logger(output_dir="logs", config=config)
    logger.info(config)
    #test_forward(config)
    #torch.autograd.set_detect_anomaly(True)
    main(config)