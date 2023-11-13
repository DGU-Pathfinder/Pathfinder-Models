from config import *
from dataset import RT_dataset
from loss import Averager
from model import get_net
from transform import *

import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    images, targets = zip(*batch)
    
    max_boxes=3
    for i, target in enumerate(targets):
        boxes=target['boxes']
        labels=target['labels']
        
        if len(boxes)<max_boxes:
            # boxes padding
            padded_boxes = torch.zeros((max_boxes, 4), dtype=boxes.dtype)
            padded_boxes[:len(boxes)] = boxes
            targets[i]['boxes'] = padded_boxes
            
            # labels padding
            padded_labels = torch.full((max_boxes,), -1, dtype=torch.int64)  # -1로 padding
            padded_labels[:len(labels)] = labels
            targets[i]['labels'] = padded_labels

    # torch.tensor로 변환
    images = torch.stack([torch.tensor(image, dtype=torch.float32) for image in images])
    boxes = [torch.tensor(target['boxes'], dtype=torch.float32) for target in targets]
    labels = [torch.tensor(target['labels'], dtype=torch.int64) for target in targets]
    
    return images, {'boxes': boxes, 'labels': labels}


def train_fn(num_epochs, train_data_loader, optimizer, model, device, clip=35):
    loss_hist = Averager()
    best_loss=float('inf')
    model.train()

    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets in tqdm(train_data_loader):
            images = images.to(device).float()
       
           
            boxes=[t.to(device) for t in targets['boxes']]
            labels=[t.to(device) for t in targets['labels']]
            
                
            
            target={"bbox":boxes,"cls":labels}
            
            loss,cls_loss,box_loss=model(images,target).values()
            loss_value=loss.detach().item()
            
            loss_hist.send(loss_value)
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            
            # grad clip
            torch.nn.utils.clip_grad_norm(model.parameters(),clip)
            
            optimizer.step()
        
        #epoch의 평균 loss 계산
        epoch_loss=loss_hist.value

        print(f'Epoch #{epoch+1} loss: {epoch_loss}')
        
        if epoch_loss<best_loss:
            best_loss=epoch_loss
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(), f'./models/effdet_best_loss_modifiedann.pth')
            print('saved best model')
         


def main():
    train_ann='/home/irteam/junghye-dcloud-dir/pathfinder/data/annotations/train.csv'
    
    data_dir='/home/irteam/junghye-dcloud-dir/pathfinder/data/Image'
   
    train_dataset=RT_dataset(data_dir,train_ann,get_train_transform())
   
    
    train_data_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    model=get_net()
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    loss = train_fn(num_epochs, train_data_loader, optimizer, model, device)
    
    
    
if __name__ == '__main__':
    main()


