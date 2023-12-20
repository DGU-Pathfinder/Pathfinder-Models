# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(train):
    
    if train:
        return A.Compose([
                            A.Resize(512,512),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                            
                            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                            A.VerticalFlip(p=0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            A.Resize(512,512),
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

