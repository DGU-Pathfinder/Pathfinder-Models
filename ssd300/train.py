import sys
sys.path.append('/content/drive/MyDrive/Pathfinder-Models/utils')
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
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models.detection.ssd as ssd
from torchvision.ops import nms

from engine import train_one_epoch,evaluate
from tqdm import tqdm
import utils
import transforms as T
from dataset import RT_Dataset
from config import Config
from augment import get_transform,get_ssd_transform
import wandb

from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights


# model
def get_object_detection_model(num_classes=4,size=300):
    # Load the Torchvision pretrained model.
    model = torchvision.models.detection.ssd300_vgg16(
        weights=SSD300_VGG16_Weights.COCO_V1
    )
    # Retrieve the list of input channels.
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Image size for transforms.
    model.transform.min_size = (size,)
    model.transform.max_size = size
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

   
        for out,target in zip(output,targets):
            scores=out['scores'].detach().cpu().numpy()
            boxes=out['boxes'].detach().cpu().numpy()
            labels=out['labels'].detach().cpu().numpy()

            keep_idx=nms(boxes,scores,iou_threshold=0.1)

            boxes=boxes[keep_idx]
            scores=scores[keep_idx]
            labels=labels[keep_idx]
            

            outputs.append({'boxes': boxes, 
                            'scores': scores,
                            'labels': labels})

            # label 포함시켜 ground truths에 추가
            gt_boxes=target['boxes'].cpu().numpy()
            gt_labels=target['labels'].cpu().numpy()

     
            ground_truths.append(list(zip(gt_labels,gt_boxes)))


    return outputs,ground_truths


def validate_and_save_best_model(epoch, model, valid_dataloader, device, optimizer, lr_scheduler, score_threshold=0.2):
    global best_recall
    outputs, ground_truths = valid_fn(valid_dataloader, model, device)
    predictions = []
    for output in outputs:
        valid_boxes = output['boxes'][output['scores'] > score_threshold]
        valid_labels=output['labels'][output['scores']> score_threshold]
        #predictions.append(valid_boxes)
        predictions.append(list(zip(valid_labels,valid_boxes)))

    # utils 모듈에 있는 calculate_metrics 함수를 사용
    metrics = utils.calculate_metrics(predictions, ground_truths)

    # 전체 성능
    total_recall=metrics['total']['recall']
    total_precision=metrics['total']['precision']
    total_f1_score= metrics['total']['f1_score']

    
    wandb.log({"epoch": epoch, "total_recall": total_recall, "total_precision": total_precision,"total_f1_score":total_f1_score})

    categories={2: 'Porosity', 3: 'Slag'}
    class_result = {class_label: metrics_val for class_label, metrics_val in metrics['per_class'].items() if class_label != 0}
    # 각 클래스별 성능 로그
    for class_label,class_metrics in metrics['per_class'].items():
      
      if class_label==2 or class_label==3:

        wandb.log({
            f"class_{categories[class_label]}_recall" : class_metrics['recall'],
            f"class_{categories[class_label]}_precision" : class_metrics['precision'],
            f"class_{categories[class_label]}_f1_score" : class_metrics['f1_score'],
            f"class_{categories[class_label]}_average_iou": class_metrics['average_iou'],

        })
    if total_recall > best_recall:
        best_recall = total_recall

        model_save_path = f"/content/drive/MyDrive/models/ssd300_real_{Config['TRAIN_BS']}_{Config['VALID_BS']}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, model_save_path)
        

    return_outputs=metrics['total']
    return return_outputs,class_result

if __name__=='__main__':
    wandb.init(project='capstone',name='SSD300_VGG16',reinit=True)

    if torch.cuda.is_available():
        device=torch.device('cuda')


    train_df=pd.read_csv('/content/drive/MyDrive/data/annotations_v2/train_total.csv')
    valid_df=pd.read_csv('/content/drive/MyDrive/data/annotations_v2/valid_total.csv')
    image_dir='/content/drive/MyDrive/data/Image'

    result_dir_path=f'/content/drive/MyDrive/result/{Config["MODEL"]}'
    os.makedirs(result_dir_path,exist_ok=True)

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

    model=get_object_detection_model(Config['NUM_CLASSES'],Config['IMG_SIZE'])
    model.to(device)

    wandb.watch(model)

    params = [p for p in model.parameters() if p.requires_grad]

    

    optimizer = torch.optim.Adam(params, lr=Config['LR'])

    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9) # 3epoch마다 학습률 10%씩 감소

    best_recall=-100
    with open(f'{result_dir_path}/{Config["TRAIN_BS"]}_{Config["VALID_BS"]}_{Config["EPOCHS"]}.txt','w') as f:
      for epoch in range(Config['EPOCHS']):
          train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)

          lr_scheduler.step()

          # valid data
          return_outputs,class_result = validate_and_save_best_model(epoch, model, valid_dataloader, device, optimizer, lr_scheduler)

          print(f'epoch : {epoch}, output : {return_outputs}')
          f.write(f"Epoch {epoch} Total result:{return_outputs}, class_result : {class_result}\n")