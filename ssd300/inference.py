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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms


from tqdm import tqdm
import utils
import transforms as T
from dataset import RT_Dataset
from config import Config
from augment import get_transform

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

        #print(f'target : {targets[0]}')
        for out,target in zip(output,targets):
            scores=out['scores'].detach().cpu().numpy()
            boxes=out['boxes'].detach().cpu().numpy()
            labels=out['labels'].detach().cpu().numpy()

            #keep_idx=nms(boxes,scores,iou_threshold=0.1)

            #boxes=boxes[keep_idx]
            #scores=scores[keep_idx]
            #labels=labels[keep_idx]


            outputs.append({'boxes': boxes, # 2중 리스트일 수도
                            'scores': scores,
                            'labels': labels})

            # ground truth 에 label 추가
            gt_boxes=target['boxes'].cpu().numpy()
            gt_labels=target['labels'].cpu().numpy()

            ground_truths.append(list(zip(gt_labels,gt_boxes)))

            #ground_truths.append(target['boxes'].cpu().numpy()) # 이중 리스트일 수도..


    return outputs,ground_truths


def validate_and_save_best_model(epoch, model, valid_dataloader, device, optimizer, lr_scheduler, score_threshold=0.2):
    global best_recall
    outputs, ground_truths = valid_fn(valid_dataloader, model, device)
    predictions = []


    for output in outputs:
        valid_scores=output['scores']>score_threshold

        valid_boxes = output['boxes'][output['scores'] > score_threshold]
        valid_labels=output['labels'][output['scores']> score_threshold]


        predictions.append(list(zip(valid_labels,valid_boxes)))



    # utils 모듈에 있는 calculate_metrics 함수를 사용
    metrics = utils.calculate_metrics(predictions, ground_truths)

    # 전체 성능
    total_recall=metrics['total']['recall']
    total_precision=metrics['total']['precision']
    total_f1_score= metrics['total']['f1_score']

    class_result = {class_label: metrics_val for class_label, metrics_val in metrics['per_class'].items() if class_label != 0}

    return_outputs=metrics['total']
    return return_outputs,class_result

if __name__=='__main__':


    if torch.cuda.is_available():
        device=torch.device('cuda')


    valid_df=pd.read_csv('/content/drive/MyDrive/data/annotations_v2/test_total.csv')
    image_dir='/content/drive/MyDrive/data/Image'




    valid_dataset=RT_Dataset(valid_df,image_dir,transforms=get_transform(train=False))



    valid_dataloader=torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=Config['VALID_BS'],
        shuffle=False,
        num_workers=Config['NUM_WORKERS'],
        collate_fn=utils.collate_fn,

    )


    model=get_object_detection_model(Config['NUM_CLASSES'])


    model_save_path = "/content/drive/MyDrive/models/fasterrcnn_resnet50_fpnv2_8_4.pth"
    saved_state=torch.load(model_save_path,map_location=device)

    model.load_state_dict(saved_state['model_state_dict'])

    model.to(device)



    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=Config['LR'], momentum=0.9, weight_decay=Config['WEIGHT_DECAY'])

    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)

          # valid data
    return_outputs,class_result = validate_and_save_best_model(0, model, valid_dataloader, device, optimizer, lr_scheduler)

    print(f'test Total result : {return_outputs} , class result : {class_result} ')