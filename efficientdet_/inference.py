import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import os
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from effdet import get_efficientdet_config,EfficientDet,DetBenchTrain
from effdet.efficientdet import HeadNet

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# RT_dataset 클래스 선언

class RT_dataset(Dataset):
    '''
    image_dir : 이미지가 존재하는 폴더 경로
    anno_path : train, valid, test 등 
    
    '''
    def __init__(self,image_dir:str,anno_path:str,transforms=None):
        super().__init__()
        self.annotations=pd.read_csv(anno_path)
        self.image_dir=image_dir
        self.transforms=transforms
        
    def __getitem__(self,idx:int):
        
        record=self.annotations.iloc[idx]
        image_name=record['image_name']
        
        image_path=os.path.join(self.image_dir,record['dataset'])
        
        image=cv2.imread(os.path.join(image_path,image_name))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image/=255.0
        
        # transform
        if self.transforms:
            sample=self.transforms(image=image)
            
           
        return sample['image'],idx
    
    
    def __len__(self):
        return len(self.annotations)
    
    
def get_test_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])
    
    
from effdet import DetBenchPredict
import gc

# Effdet config를 통해 모델 불러오기 + ckpt load
def load_net(checkpoint_path, device):
    config = get_efficientdet_config('tf_efficientdet_d3')
    config.num_classes = 3
    config.image_size = (512,512)
    
    config.soft_nms = False
    config.max_det_per_image = 25
    
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    net = DetBenchPredict(net)
    net.load_state_dict(checkpoint)
    net.eval()

    return net.to(device)


# valid function
def valid_fn(val_data_loader,test_ann, model, device):
    outputs = []
    ground_truths=[]
    for images, image_ids in tqdm(val_data_loader):
        # gpu 계산을 위해 image.to(device)       
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()
        print(image_ids)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4], 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]})
        for image_id in image_ids:
            record=test_ann.iloc[image_id]
            boxes=np.array([bbox for bbox in eval(record['bndbox'])])
            ground_truths.append(boxes)
            
        
    return outputs,ground_truths

def collate_fn(batch):
    return tuple(zip(*batch))


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
def calculate_metrics(predictions, ground_truths, iou_threshold=0.0001):
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
    test_ann_dir='/home/irteam/junghye-dcloud-dir/pathfinder/data/annotations/val.csv'
    image_dir='/home/irteam/junghye-dcloud-dir/pathfinder/data/Image'
    test_dataset=RT_dataset(image_dir,test_ann_dir,transforms=get_test_transform())
    test_ann=pd.read_csv(test_ann_dir)
    #epoch=50
    checkpoint_path=f'/home/irteam/junghye-dcloud-dir/pathfinder/models/effdet_best_loss_modifiedann.pth'
    score_threshold=0.01 # score : 모델이 해당 객체를 올바르게 감지했다고 확신하는 정도  
    test_data_loader=DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    model=load_net(checkpoint_path,device)

    outputs,ground_truths=valid_fn(test_data_loader,test_ann,model,device)
    
    # calculate precision, recall, average_iou scores 
    # 테스트 데이터에서 상위 점수를 가진 bounding box만 선택
    predictions = []
    for output in outputs:
        valid_boxes = output['boxes'][output['scores'] > score_threshold]
        predictions.append(valid_boxes)

    # calculate precision, recall, average_iou scores
    metrics = calculate_metrics(predictions, ground_truths)
    print(metrics)
    
if __name__ == "__main__":
    main()