import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

import torch
import torch.nn as nn

from utils import outputs_to_masks
from timm.utils import AverageMeter

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    loss_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    print("start validate")
    end = time.time()
    for idx, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(config["device"])
        labels = labels.to(config["device"])
                
        with torch.cuda.amp.autocast(enabled=config["use_amp"]):
            outputs = model(imgs)
        
        loss = criterion(outputs, labels)        
        outputs = outputs_to_masks(outputs)
        labels = labels.cpu().numpy()
        print(outputs.shape, labels.shape)
        precision, recall, f1_score, iou = eval_metrics(outputs, labels)
        
        loss_meter.update(loss.item(), config["batch_size"])
        precision_meter.update(precision, config["batch_size"])
        recall_meter.update(recall, config["batch_size"])
        f1_meter.update(f1_score, config["batch_size"])
        iou_meter.update(iou, config["batch_size"]) 
        
        end = time.time()
        
        #if idx % print_freq == 0:
        #    pass
        
    return precision_meter.avg, recall_meter.avg, f1_meter.avg, iou_meter.avg, loss_meter.avg
        
# precision, recall, F1 score, IoU
def eval_metrics(outputs, targets):
    # Flatten the arrays
    #batch_size = outputs.shape(0)
    #outputs = outputs.reshape(batch_size, -1)
    #targets = targets.reshape(batch_size, -1)
    #outputs = outputs.flatten()
    #targets = targets.flatten()
        
    # Calculate precision, recall, F1 score and intersection over union (IoU)
    precision = precision_score(targets, outputs, average="micro")
    recall = recall_score(targets, outputs, average="micro")
    f1 = f1_score(targets, outputs, average="micro")
    iou = jaccard_score(targets, outputs, average="micro")   
    
    return precision, recall, f1, iou
        
        
    