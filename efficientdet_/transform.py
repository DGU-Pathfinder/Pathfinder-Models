import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_train_transform():
    return A.Compose([
        A.Resize(512,512),
        A.Flip(p=0.5),
        A.RandomBrightness(limit=0.24,p=0.5),
        ToTensorV2(p=1.0)
    ],bbox_params={'format':'pascal_voc','label_fields':['labels']})
    
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ],bbox_params={'format':'pascal_voc','label_fields':['labels']})
    