# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(train):
    
    if train:
        return A.Compose([
            
                            A.HorizontalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.VerticalFlip(p=0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})