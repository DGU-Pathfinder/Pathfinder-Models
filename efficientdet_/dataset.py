import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import os
import numpy as np

import cv2



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
        
        # boxes(xmin,ymin,xmax,ymax)
        boxes=np.array([bbox for bbox in eval(record['bndbox'])])
        labels=np.array(eval(record['labels']))
        
        if len(boxes)==0:
          
            boxes=np.array([[0,0,1,1]],dtype=np.float32)
           
            labels=np.array([0],dtype=np.int64)
           
       
        
        target={'boxes':boxes,'labels':np.array(labels)}

        
        # transform
        if self.transforms:
            
            sample=self.transforms(**{
                    'image':image,
                    'bboxes':target['boxes'],
                    'labels':target['labels']
            })
            image=sample['image']
            
            if len(sample['bboxes'])>0:
                boxes=np.array(sample['bboxes'])
                boxes = boxes[:, [1, 0, 3, 2]]
                target['boxes']=boxes
                target['labels']=np.array(sample['labels'])
            
        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
                    
        return image, target
    
    def __len__(self):
        return len(self.annotations)