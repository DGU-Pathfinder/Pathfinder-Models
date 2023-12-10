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
from torchvision.ops import nms

from engine import train_one_epoch,evaluate
from tqdm import tqdm
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
    
# valid function
def valid_fn(val_data_loader, model, device):
    model.eval()
    outputs = []
    ground_truths=[]
    
    for images,targets in tqdm(val_data_loader):
       
        images=list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        output=model(images)
    
        #print(f'target : {targets[0]}')
        for out,target in zip(output,targets):
            scores=out['scores'].detach().cpu()
            boxes=out['boxes'].detach().cpu()
            labels=out['labels'].detach().cpu()
            
            keep_idx=nms(boxes,scores,iou_threshold=0.1)

            boxes=boxes[keep_idx]
            scores=scores[keep_idx]
            labels=labels[keep_idx]

            outputs.append({'boxes': boxes, # 2중 리스트일 수도
                            'scores': scores, 
                            'labels': labels})
           
            ground_truths.append(target['boxes'].cpu().numpy()) # 이중 리스트일 수도.. 

        
    return outputs,ground_truths


def validate_and_save_best_model(epoch, model, valid_dataloader, device, optimizer, lr_scheduler,score_threshold=0.01):
    global best_recall  
    outputs, ground_truths = valid_fn(valid_dataloader, model, device)
    predictions = []
    for output in outputs:
        valid_boxes = output['boxes'][output['scores'] > score_threshold]
        predictions.append(valid_boxes)

    # utils 모듈에 있는 calculate_metrics 함수를 사용합니다.
    # calculate_metrics 함수가 올바르게 구현되어 있어야 합니다.
    metrics = utils.calculate_metrics(predictions, ground_truths)
    #wandb.log({"epoch": epoch, "recall": metrics['recall']})  # Recall을 W&B에 로그합니다.

    if metrics['recall'] > best_recall:
        best_recall = metrics['recall']
        model_save_path = f"./models/faster-rcnn_finetuned_resize512_bestrecall.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, model_save_path)
        #wandb.save(model_save_path)  # 모델 파일을 W&B에 저장합니다.

    return metrics

if __name__=='__main__':
    wandb.init(project='capstone',name='faster-rcnn-best-recall',reinit=True)
    
    if torch.cuda.is_available():
        device=torch.device('cuda')
        
    
    train_df=pd.read_csv('/home/irteam/junghye-dcloud-dir/pathfinder/pathfinder_ai/annotations_v2/train_total.csv')
    valid_df=pd.read_csv('/home/irteam/junghye-dcloud-dir/pathfinder/pathfinder_ai/annotations_v2/valid_total.csv')
    image_dir='/home/irteam/junghye-dcloud-dir/pathfinder/data_contrast/before/Image'

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
    
    best_recall=-100
    for epoch in range(Config['EPOCHS']):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        
        lr_scheduler.step()

        # valid data
        metrics = validate_and_save_best_model(epoch, model, valid_dataloader, device, optimizer, lr_scheduler)
        print(f"Epoch {epoch} Metrics:", metrics)
        
        #txt_path='./faster-rcnn-output.txt'
        #evaluate(model, valid_dataloader, device=device,file_path=txt_path)



