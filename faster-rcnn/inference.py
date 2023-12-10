import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

import os
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# load module
from dataset import RT_Dataset
from augment import *
from config import Config
import utils


# valid function
def valid_fn(val_data_loader, model, device):
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

def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model


# IoU 계산 함수
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    
    # Calculate area of intersection
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    intersection_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # Calculate area of union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou

# 메트릭 계산 함수
def calculate_metrics(predictions, ground_truths, iou_threshold=0.2):
    """
    Calculate precision, recall, and IoU score for a set of predictions and ground truth boxes.
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_iou_score = 0
    
    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        matched = [False] * len(gt_boxes)
        
        for pred_box in pred_boxes: # 모든 prediction에 대해 
            best_iou = 0
            best_match = None
            
            for i, gt_box in enumerate(gt_boxes):
                #if i==0:
                    #print(f'gt : {gt_box[0][0]}')
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_iou > iou_threshold:
                if not matched[best_match]:
                    total_true_positives += 1
                    total_iou_score += best_iou
                    matched[best_match] = True
            else:
                total_false_positives += 1
        
        total_false_negatives += len(gt_boxes) - sum(matched)
    
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) != 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) != 0 else 0
    average_iou = total_iou_score / total_true_positives if total_true_positives != 0 else 0
    
    return {'precision': precision, 'recall': recall, 'average_iou': average_iou}


def main():
    test_ann_path='/home/irteam/junghye-dcloud-dir/pathfinder/pathfinder_ai/annotations_v2/test_total.csv'
    test_ann=pd.read_csv(test_ann_path)
    
    image_dir='/home/irteam/junghye-dcloud-dir/pathfinder/data_contrast/before/Image'
    
    
    test_dataset=RT_Dataset(test_ann,image_dir,transforms=get_transform(train=False))
    test_data_loader=DataLoader(
        test_dataset,
        batch_size=Config['VALID_BS'],
        shuffle=False,
        num_workers=Config['NUM_WORKERS'],
        collate_fn=utils.collate_fn,
    )
    
    model=get_object_detection_model(Config['NUM_CLASSES'])
   
    checkpoint=torch.load('/home/irteam/junghye-dcloud-dir/pathfinder/pathfinder_ai/models/faster-rcnn_finetuned_resize.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    

    score_threshold=0.01 # score : 모델이 해당 객체를 올바르게 감지했다고 확신하는 정도  
    
    
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
 
    model.eval()
    model.to(device)

    outputs,ground_truths=valid_fn(test_data_loader,model,device)
    
    # calculate precision, recall, average_iou scores 
    # 테스트 데이터에서 상위 점수를 가진 bounding box만 선택
    predictions = []
    for output in outputs:
        valid_boxes = output['boxes'][output['scores'] > score_threshold]
        predictions.append(valid_boxes)

    metrics = calculate_metrics(predictions, ground_truths)
    print(metrics)
    
if __name__ == "__main__":
    main()