import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import visualize_segmentation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DaclDataset(Dataset):
    def __init__(self, img_dir, json_dir, num_classes = 19, transform=None):
        """
        Args:
            img_dir (str): Directory containing the images.
            json_dir (str): Directory containing the JSON annotation files.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_size (tuple): Desired size of the images.
        """
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.num_classes = num_classes
        self.transform = transform
        

        self.class_map = {  # A mapping from the label to index
                "Crack": 0,
                "ACrack": 1,
                "Wetspot": 2,
                "Efflorescence": 3,
                "Rust": 4,
                "Rockpocket": 5,
                "Hollowareas": 6,
                "Cavity": 7,
                "Spalling": 8,
                "Graffiti": 9,
                "Weathering": 10,
                "Restformwork": 11,
                "ExposedRebars": 12,
                "Bearing": 13,
                "EJoint": 14,
                "Drainage": 15,
                "PEquipment": 16,
                "JTape": 17,
                "WConccor": 18
        }
        
        # Get the list of image files (assumes JSON and image files have the same base name)
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.annotations_files = [f for f in os.listdir(json_dir) if f.endswith('.json')] 

    def __len__(self):
        return len(self.image_files)

    
    def __getitem__(self, idx):
    
        json_file = self.annotations_files[idx]
        json_path = os.path.join(self.json_dir, json_file)
        
        with open(json_path, 'r') as f:
            ann = json.load(f)

        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        # Create the segmentation mask for the current image
        mask = np.zeros((image.shape[0], image.shape[1], len(self.class_map)), dtype=np.float32) #CHECK CHANGE HERE
        
        for shape in ann['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            
            # Get the index for the current label
            if label in self.class_map:
                label_idx = self.class_map[label]
    
                channel_mask = mask[:, :, label_idx].copy()
                cv2.fillPoly(channel_mask, [points], 1)
                mask[:, :, label_idx] = channel_mask


        if self.transform:
            # print(f"Image shape:{image.shape}, Mask shape: {mask.shape}")
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # print(f"Image shape:{image.shape}, Mask shape: {mask.shape}")
            # print(image.dtype, mask.dtype)
        return image, mask


train_transform = A.Compose(
    [
    A.Resize(height=640,width=640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    #A.RandomRotate90(p=1.0),
    # A.Affine(shift_limit=0.07, scale_limit=0.07, rotate_limit=10, border_mode=cv2.BORDER_REFLECT, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3,contrast_limit=0.3,p=0.5),
    A.MotionBlur(p=0.5),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), fill=0, fill_mask=0, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],normalization="standard"), #image only
    ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
    A.Resize(height=640,width=640),
    A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #image only
    ToTensorV2(),
    ]
)

img_dir = './images/'  
json_dir = './annotations/'  

train_dataset = DaclDataset(img_dir + "/train", json_dir + "/train", transform=train_transform)
validation_dataset = DaclDataset(img_dir + "/validation", json_dir + "/validation", transform=val_transform)

image, mask = train_dataset[0]  # Both are tensors
# validate_mask(mask, image.shape, num_classes=19)
image = image.float().unsqueeze(0)
mask = mask.unsqueeze(0) 
# print(f"Image shape:{image.shape}, Mask shape: {mask.shape}")
# visualize_segmentation(validation_dataset, idx=827, samples=3)