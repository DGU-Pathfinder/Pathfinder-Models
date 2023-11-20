import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np


class RT_Dataset(Dataset):
    def __init__(self,dataframe,image_dir,transforms=None):
        super().__init__()
        self.image_ids=dataframe['image_number'].unique()
        self.df=dataframe
        self.image_dir=image_dir
        self.transforms=transforms
        
    def __getitem__(self,index:int):
        image_id=self.image_ids[index]
        records=self.df[self.df['image_number']==image_id]
        
        image=cv2.imread(f'{self.image_dir}/{records["dataset"].values[0]}/{records["image_name"].values[0]}',cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image/=255.0
        
        bndbox=np.array(eval(records['bndbox'].values[0]))
        labels=np.array(eval(records['labels'].values[0]))
        boxes=np.zeros((bndbox.shape[0],5))
        
        if bndbox.size>0:
            boxes[:,0:4]=bndbox
            boxes[:,4]=labels
        
        sample={'img':image, 'annot':boxes}
        
        if self.transforms:
            sample=self.transforms(sample)
            
        return sample
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]