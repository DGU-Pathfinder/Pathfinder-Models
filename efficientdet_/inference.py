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
    config.num_classes = 4
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
    labels=[]
    
    for images, image_ids in tqdm(val_data_loader):
        # gpu 계산을 위해 image.to(device)       
        image_dir='./data_test/01. data/Image'
        record=test_ann.iloc[image_ids]
        image_name=record['image_name']
        image_path=os.path.join(image_dir,record['dataset'])
        test=cv2.imread(os.path.join(image_path,image_name))
        h, w, c = test.shape
        
        a=images
        images = torch.stack(images) # bs, ch, w, h 
        images = images.to(device).float()

        output = model(images)
        
        for out in output:
            outputs.append({'boxes': out.detach().cpu().numpy()[:,:4]*np.array([w,h,w,h])/512, 
                            'scores': out.detach().cpu().numpy()[:,4], 
                            'labels': out.detach().cpu().numpy()[:,-1]})
        for image_id in image_ids:
            record=test_ann.iloc[image_id]
            boxes=np.array([bbox for bbox in eval(record['bndbox'])])
            ground_truths.append(boxes)
            
        
    return outputs,ground_truths,labels

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
def calculate_metrics(predictions_boxes,predictions_labels, ground_truths,labels, iou_threshold=0.2):
    """
    Calculate precision, recall, and IoU score for a set of predictions and ground truth boxes.
    """
    result=np.zeros((4,4))

    for pred_boxes, pred_labels, gt_boxes,gt_labels in zip(predictions_boxes, predictions_labels, ground_truths, labels):
        if len(pred_labels)==0:
            pred_labels=np.append(pred_labels,3)
            pred_boxes=np.append(pred_boxes,0)

        for pred_box,pred_label in zip(pred_boxes,pred_labels): # 모든 prediction에 대해 
            best_iou = 0
            for gt_box, gt_label in zip(gt_boxes,gt_labels):
                if pred_label==3:
                    x=int(pred_label)
                else:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        x=int(gt_label)
                        y=int(pred_label)
            
            if best_iou > iou_threshold :
                result[x][y]+=1

            else:
                if len(gt_boxes)==0:
                    result[3][int(pred_label)]+=1
                else:
                    result[x][3]+=1

    print(result)
    Porosity_precision = result[1][1]/sum(result[:,1])
    Porosity_recall = result[1][1]/sum(result[1,:])
    Porosity_F1_score = 2*Porosity_precision*Porosity_recall/(Porosity_precision+Porosity_recall)
    slag_precision = result[2][2]/sum(result[:,2])
    slag_recall = result[2][2]/sum(result[2,:])
    slag_F1_score = 2*slag_precision*slag_recall/(slag_precision+slag_recall)

    return {'Porosity_precision': Porosity_precision, 'Porosity_recall': Porosity_recall, 'Porosity_F1_score': Porosity_F1_score,
           'slag_precision': slag_precision, 'slag_recall': slag_recall, 'slag_F1_score': slag_F1_score}


def main():
    test_ann_dir='./data_test/01. data/annotations_v2/test_total.csv'
    image_dir='./data_test/01. data/Image'
    test_dataset=RT_dataset(image_dir,test_ann_dir,transforms=get_test_transform())
    test_ann=pd.read_csv(test_ann_dir)
    #epoch=50
    checkpoint_path=f'/home/irteam/junghye-dcloud-dir/pathfinder/models/effdet_best_loss_modifiedann.pth'
    score_threshold=0.2 # score : 모델이 해당 객체를 올바르게 감지했다고 확신하는 정도  
    test_data_loader=DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    model=load_net(checkpoint_path,device)

    outputs,gt_boxes,gt_labels=valid_fn(test_data_loader,test_ann,model,device)
    
    # calculate precision, recall, average_iou scores 
    # 테스트 데이터에서 상위 점수를 가진 bounding box만 선택
    predictions = []
    for output in outputs:
        valid_boxes = output['boxes'][output['scores'] > score_threshold]
        predictions.append(valid_boxes)

    # calculate precision, recall, average_iou scores
    metrics = calculate_metrics(predictions_boxes,predictions_labels,gt_boxes,gt_labels)
    print(metrics)
    
if __name__ == "__main__":
    main()
