import os
import numpy as np
import pandas as pd

# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

import cv2

import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch,evaluate
import utils
import transforms as T
from dataset import RT_Dataset
from config import Config
from augment import get_transform
import wandb


# model

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model
    



if __name__=='__main__':
    wandb.init(project='capstone',name='faster-rcnn',reinit=True)
    
    if torch.cuda.is_available():
        device=torch.device('cuda')
        
    
    train_df=pd.read_csv('../../data/annotations/train_total.csv')
    valid_df=pd.read_csv('../../data/annotations/valid_total.csv')
    image_dir='../../data_contrast/after/Image'

    train_dataset=RT_Dataset(train_df,image_dir,transforms=get_transform(train=True))
    valid_dataset=RT_Dataset(valid_df,image_dir,transforms=get_transform(train=False))

    train_dataloader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config['TRAIN_BS'],
        shuffle=True,
        num_workers=Config['NUM_WORKERS'],
        collate_fn=utils.collate_fn,
    )

    valid_dataloader=torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=Config['VALID_BS'],
        shuffle=False,
        num_workers=Config['NUM_WORKERS'],
        collate_fn=utils.collate_fn,
        
    )

    model=get_object_detection_model(Config['NUM_CLASSES'])
    model.to(device)
    
    wandb.watch(model)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, lr=Config['LR'], momentum=0.9, weight_decay=Config['WEIGHT_DECAY'])
    
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    
    
    for epoch in range(Config['EPOCHS']):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        
        lr_scheduler.step()
        
        evaluate(model, valid_dataloader, device=device)



