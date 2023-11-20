import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from config import Config

class RT_Dataset(Dataset):
    def __init__(self,dataframe,image_dir,transforms=None):
        super().__init__()
        self.image_ids=dataframe['image_number'].unique()
        self.df=dataframe
        self.image_dir=image_dir
        self.transforms=transforms
        #self.classes=[_,'Others','Porosity','Slag']
        
        
    def __getitem__(self,index:int):
        image_id=self.image_ids[index]
        records=self.df[self.df['image_number']==image_id]
        
        image=cv2.imread(f'{self.image_dir}/{records["dataset"].values[0]}/{records["image_name"].values[0]}',cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        #image_res=cv2.resize(image_rgb,(Config['IMG_SIZE'],Config['IMG_SIZE']),cv2.INTER_AREA)
        image /=255.0
        
        #print(f'image : {type(image)}')
        labels=[]
        
        wt,ht=image.shape[1],image.shape[0]
        
        bndboxes=list((eval(records['bndbox'].values[0])))
        labels=list(eval(records['labels'].values[0]))        

        if len(bndboxes)>0: 
            boxes = [[box[0] , box[1], box[2], box[3]] for box in bndboxes]
            labels=[int(label)+1 for label in labels]
            boxes=torch.as_tensor(boxes,dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        else:
            boxes = torch.zeros((0,4),dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)  # 더미 라벨
            area = torch.zeros(0, dtype=torch.float32)  # 더미 면적
            
        
        #다 crowd x
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        
        
        target={}
        target['boxes']=boxes
        target['labels']=labels
        target['area']=area
        target['iscrowd']=iscrowd
        target['image_id']=torch.tensor([image_id])
        
        if self.transforms:
            sample={
                'image':image,
                'bboxes':target['boxes'],
                'labels':labels
            }
            
            #target['boxes']=torch.Tensor(sample['bboxes'])
            sample=self.transforms(**sample)
            image=sample['image']
            
            if len(sample['bboxes'])>0:
                target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)
            else:
                target['boxes']=torch.zeros((0,4),dtype=torch.float32)
        
        return image,target 
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]