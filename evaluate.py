import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

import torch
import torch.nn as nn

from utils import outputs_to_masks
from timm.utils import AverageMeter

@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    loss_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    start_time = time.time()
    for idx, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(config["device"])
        labels = labels.to(config["device"])
                
        #with torch.cuda.amp.autocast(enabled=config["use_amp"]):
        outputs = model(imgs)
        
        loss = criterion(outputs, labels)        
        outputs = outputs_to_masks(outputs)
        labels = labels.cpu().numpy()
        # batch size * 512 * 512
        #print(outputs.shape, labels.shape) 
        precision, recall, f1_score, iou = eval_metrics(outputs, labels)
        
        loss_meter.update(loss.item(), config["batch_size"])
        precision_meter.update(precision, config["batch_size"])
        recall_meter.update(recall, config["batch_size"])
        f1_meter.update(f1_score, config["batch_size"])
        iou_meter.update(iou, config["batch_size"]) 
        
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f'Validate\t'
        f'Time {time.time() - start_time}\t'
        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'precision {precision_meter.val:.4f} ({precision_meter.avg:.4f})\t'
        f'recall {recall_meter.val:.4f} ({recall_meter.avg:.4f})\t'
        f'f1 score {f1_meter.val:.4f} ({f1_meter.avg:.4f})\t'
        f'iou {iou_meter.val:.4f} ({iou_meter.avg:.4f})\t'
        f'mem {memory_used:.0f}MB')
        
    return precision_meter.avg, recall_meter.avg, f1_meter.avg, iou_meter.avg, loss_meter.avg
        
# precision, recall, F1 score, IoU
def eval_metrics(outputs, targets):
    # Flatten the arrays
    #batch_size = outputs.shape(0)
    #outputs = outputs.reshape(batch_size, -1)
    #targets = targets.reshape(batch_size, -1)
    outputs = outputs.flatten()
    targets = targets.flatten()
        
    # Calculate precision, recall, F1 score and intersection over union (IoU)
    precision = precision_score(targets, outputs, average="micro")
    recall = recall_score(targets, outputs, average="micro")
    f1 = f1_score(targets, outputs, average="micro")
    iou = jaccard_score(targets, outputs, average="micro")   
    
    return precision, recall, f1, iou
        
        
    