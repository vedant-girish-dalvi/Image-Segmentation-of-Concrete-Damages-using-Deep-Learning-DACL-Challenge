import os
import random
import cv2
import torch
import numpy as np
from utils import load_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from utils import visualize_mask

# ----------------------------- Settings -----------------------------
IMG_DIR = "./images/validation"
MODEL_PATH = "./checkpoints/best_model_27e_0.27val_0.51tra.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 19
IMAGE_SIZE = (640, 640)  # (H, W) based on your training setup
SAVE_PATH = "./inference_output"
os.makedirs(SAVE_PATH, exist_ok=True)

# ----------------------------- Transform -----------------------------
transform = A.Compose([
    A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ----------------------------- Load Model -----------------------------
model = load_model(MODEL_PATH, device=DEVICE)

# ----------------------------- Run Inference -----------------------------

img_files = [f for f in os.listdir(IMG_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
img_file = random.choice(img_files)
img_path = os.path.join(IMG_DIR, img_file)
original_img = cv2.imread(img_path)
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
augmented = transform(image=original_img_rgb)
img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    output = model(img_tensor)
    output = torch.sigmoid(output.squeeze(0))  # Shape: (C, H, W)

# Visualize and save
pred_mask = visualize_mask(output)
original_img_resized = cv2.resize(original_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))  # (W, H)
overlay = cv2.addWeighted(original_img_resized, 0.6, pred_mask, 0.4, 0)

cv2.imwrite(os.path.join(SAVE_PATH, f"mask_{img_file}"), pred_mask)
cv2.imwrite(os.path.join(SAVE_PATH, f"overlay_{img_file}"), overlay)

print(f"Inference done. Saved mask and overlay for: {img_file}")
