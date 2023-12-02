import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import re
import cv2
import time
import numpy as np
import pandas as pd


import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader, Dataset

from retinanet import model
from retinanet.dataloader import collater, Resizer, Augmenter, Normalizer, UnNormalizer,CSVDataset,AspectRatioBasedSampler
from config import Config
from dataset import RT_Dataset
import wandb
    
        

best_loss=float('inf')

def train_one_epoch(retinanet,epoch_num,optimizer,scheduler, train_data_loader,device):
    
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    retinanet.train()
    
    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):
                
        # Reseting gradients after each iter
        optimizer.zero_grad()
            
        # Forward
        #print(f'{data["annot"]}')
        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])
        
        # Calculating Loss
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss

        if bool(loss == 0):
            continue
                
        # Calculating Gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                
        # Updating Weights
        optimizer.step()

        #Epoch Loss
        epoch_loss.append(float(loss))
        
        print(
            'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

        del classification_loss
        del regression_loss
        
    wandb.log({
        'avg_train_loss':round(np.mean(epoch_loss),4)
    })
    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step(np.mean(epoch_loss))
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    
    
def valid_one_epoch(retinanet,epoch_num, valid_data_loader,device):
    global best_loss #가장 좋은 손실값 업데이트하기 위해 global 변수로 선언
    
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()
    
    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):
                
        with torch.no_grad():
            
            # Forward
            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

            # Calculating Loss
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            #Epoch Loss
            epoch_loss.append(float(loss))
            
            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

            del classification_loss
            del regression_loss
        
    avg_epoch_loss=np.mean(epoch_loss)
    wandb.log({
        'avg_val_loss': round(avg_epoch_loss,4)
    })
    
    if avg_epoch_loss <best_loss:
        best_loss=avg_epoch_loss
        
        if not os.path.exists('./models'):
            os.makedirs("./models")
            
        torch.save(retinanet, f"./models/retinanet_{Config['CONTRAST']}_{Config['EPOCHS']}.pt")
        print(f"Epoch {epoch_num}: Validation loss improved, model saved.")
         
        
    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))
    
    # Save Model after each epoch
  
            
     
if __name__=='__main__':
    # wandb project
    wandb.init(project='capstone',name='retinanet_1123',reinit=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    
    # dataset 
    #train_df=pd.read_csv('/home/irteam/junghye-dcloud-dir/pathfinder/pathfinder_ai/annotations_v2/train_total.csv')
    #valid_df=pd.read_csv('/home/irteam/junghye-dcloud-dir/pathfinder/pathfinder_ai/annotations_v2/valid_total.csv')
    #image_dir='/home/irteam/junghye-dcloud-dir/pathfinder/data_contrast/before/Image'

    #train_dataset=RT_Dataset(train_df,image_dir,transforms=T.Compose([Augmenter(),Normalizer(),Resizer()]))
    #valid_dataset=RT_Dataset(valid_df,image_dir,transforms=T.Compose([Normalizer(),Resizer()]))

    # csv dataset
    
    train_dataset=CSVDataset('./annotations_v2/retinanet_train.csv','./annotations_v2/classes.csv',transform=T.Compose([Augmenter(),Normalizer()]))
    valid_dataset=CSVDataset('./annotations_v2/retinanet_valid.csv','./annotations_v2/classes.csv',transform=T.Compose([Normalizer()]))
            
    sampler=AspectRatioBasedSampler(train_dataset,batch_size=Config['TRAIN_BS'],drop_last=False)
    sampler_val=AspectRatioBasedSampler(valid_dataset,batch_size=Config['VALID_BS'],drop_last=False)
    
    
    train_data_loader = DataLoader(
        train_dataset,
        num_workers = Config['NUM_WORKERS'],
        collate_fn = collater,
        batch_sampler=sampler,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        num_workers = Config['NUM_WORKERS'],
        collate_fn = collater,
        batch_sampler=sampler_val,
    )
    # load model
    retinanet = model.resnet50(num_classes = Config['NUM_CLASSES'], pretrained = True)
    retinanet.to(device)
    wandb.watch(retinanet)
    
    
    #optimizer = torch.optim.Adam(retinanet.parameters(), lr = Config['LR'],weight_decay=Config['WEIGHT_DECAY'])
    optimizer=torch.optim.Adam(retinanet.parameters(),lr=Config['LR'])
    
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,verbose=True)

    ### Training Loop
    for epoch in range(Config['EPOCHS']):
        
        # Call train function
        train_one_epoch(retinanet,epoch, optimizer,lr_scheduler,train_data_loader,device=device)
        # Call valid function
        valid_one_epoch(retinanet,epoch, valid_data_loader,device=device)
    