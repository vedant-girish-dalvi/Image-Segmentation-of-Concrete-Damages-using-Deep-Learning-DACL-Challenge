import torch
from torch.utils.data import DataLoader
import json
import os, random
from pathlib import Path
from model import SegmentationModel
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 19
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "UnetPlusPlus_timm-efficientnet-b5_epoch20_20251019_083628.pth")
OUTPUT_DIR = "./submission_masks"
# Define label names (change as per your class labels)
CLASS_LABELS = [
        "Crack",
        "ACrack",
        "Wetspot",
        "Efflorescence",
        "Rust",
        "Rockpocket",
        "Hollowareas",
        "Cavity",
        "Spalling",
        "Graffiti",
        "Weathering",
        "Restformwork",
        "ExposedRebars",
        "Bearing",
        "EJoint",
        "Drainage",
        "PEquipment",
        "JTape",
        "WConnor"
                ]
assert len(CLASS_LABELS) == NUM_CLASSES, "Mismatch in number of class labels!"

'''Dacl dataset Class UTILS'''
# Generate a colormap for 19 classes
def generate_colormap(num_classes=19):
    np.random.seed(42)  # For consistent colors
    return np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

# Updated overlay function for multilabel masks
def overlay_mask(image, mask, alpha=0.5):
    """
    image: HWC format
    mask: (H, W, 19), dtype=bool or int
    """
    
    COLORMAP = generate_colormap()
    mask_overlay = np.zeros_like(image, dtype=np.uint8)

    for class_idx in range(mask.shape[2]):
        class_mask = mask[:, :, class_idx]
        color = COLORMAP[class_idx]
        mask_overlay[class_mask > 0] = np.clip(mask_overlay[class_mask > 0] + color, 0, 255)

    blended = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)
    return blended

def visualize_segmentation(dataset, idx=0, samples=3):
    # Strip Normalize/ToTensorV2 for visualization
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform

    figure, ax = plt.subplots(samples + 1, 2, figsize=(10, 5 * (samples + 1)))

    # Get original image and mask
    original_transform = dataset.transform
    dataset.transform = None
    image, mask = dataset[idx]
    # image = image.numpy().transpose(1, 2, 0)
    # mask = mask.numpy()  # Shape: (H, W, 19)
    dataset.transform = original_transform

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(overlay_mask(image, mask))
    ax[0, 1].set_title("Original Overlay")
    ax[0, 1].axis("off")

    for i in range(samples):
        if vis_transform:
            augmented = vis_transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
        else:
            aug_image, aug_mask = image, mask

        ax[i + 1, 0].imshow(aug_image)
        ax[i + 1, 0].set_title(f"Augmented Image {i + 1}")
        ax[i + 1, 0].axis("off")

        ax[i + 1, 1].imshow(overlay_mask(aug_image, aug_mask))
        ax[i + 1, 1].set_title(f"Augmented Overlay {i + 1}")
        ax[i + 1, 1].axis("off")

    plt.tight_layout()
    plt.show()

def validate_mask(mask, image_shape, num_classes):
    # Check mask dimensions
    # assert mask.shape[:2] == image_shape, f"Mask dimensions should match image dimensions {image_shape}"
    # assert mask.shape[2] == num_classes, f"Mask should have {num_classes} channels"

    # Check for binary values (0 or 1)
    unique_values = np.unique(mask)
    print(f"Unique values in mask: {unique_values}")
    assert set(unique_values) <= {0, 1}, "Mask should only contain 0 or 1 values"

    # Check class index validity
    # assert np.all(mask <= num_classes), "Mask class index is out of bounds"

    # Check for empty class masks (i.e., each class should have at least one non-zero pixel)
    non_zero_pixel_counts = [np.count_nonzero(mask[:, :, i]) for i in range(num_classes)]
    print(f"non zero pixel counts:{non_zero_pixel_counts}")
    assert all(count > 0 for count in non_zero_pixel_counts), "Some class masks are empty"

    print("Mask is valid!")

def get_class_mapping():
    """
    Returns a dictionary mapping class indices to their corresponding labels.
    Modify this dictionary based on your dataset's class definitions.
    """
    class_mapping = {
        0: "Background",  # Assuming 0 is the background (ignore this during predictions)
        1: "Spalling",
        2: "PEquipment",
        3: "Crack",
        4: "Rebar Exposure",
        5: "Efflorescence",
        6: "Delamination",
        7: "Corrosion",
        8: "Scaling",
        9: "Staining",
        10: "Leakage",
        11: "Debonding",
        12: "Discoloration",
        13: "Erosion",
        14: "Fracture",
        15: "Honeycombing",
        16: "Peeling Paint",
        17: "Surface Roughness",
        18: "Vegetation Growth"
    }
    return class_mapping


'''Checkpointing UTILS'''

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
import torch


def save_checkpoint(model, optimizer, epoch, filepath, scheduler=None, scaler=None, best=False):
    """
    Saves model, optimizer, scheduler, and scaler states in one checkpoint file.
    Example filename: 'checkpoints/DeepLabV3Plus_mit_b5_e055_miou0.3744_vloss0.1836.pth'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()   

    torch.save(checkpoint, filepath)

    if best:
        print(f"Saved best model checkpoint to: {filepath}")
    else:
        print(f"Saved checkpoint to: {filepath}")


def resume_from_checkpoint(model, optimizer, scheduler=None, scaler=None, filepath=None, device="cuda"):
    """
    Loads model, optimizer, scheduler, and scaler states from a checkpoint.
    Moves optimizer state tensors to the correct device (fixes CUDA/CPU mismatch).
    Returns next epoch number for training continuation.
    """
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    print(f"Loading checkpoint from: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    # --- Load model weights ---
    model.load_state_dict(checkpoint["model_state_dict"])

    # --- Load optimizer and move its state tensors to correct device ---
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # --- Load AMP scaler (if present) ---
    if scaler is not None and "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as e:
            print(f"Warning: Could not load scaler state. ({e})")

    # --- Load scheduler (if present) ---
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            print(f"Warning: Could not load scheduler state. ({e})")

    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Loaded checkpoint '{filepath}' (resuming from epoch {start_epoch})")
    return start_epoch



# def save_checkpoint(model, optimizer, epoch, filepath, best=False):
#     """
#     Saves model checkpoint.

#     Args:
#         model: torch.nn.Module
#         optimizer: torch optimizer
#         epoch: current epoch
#         filepath: path to save checkpoint
#         best: if True, mark as best model
#     """
#     state = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#     }

#     torch.save(state, filepath)
#     print(f"Checkpoint saved: {filepath}")

#     # Optionally, maintain a separate copy of the best model
#     if best:
#         best_path = os.path.join(os.path.dirname(filepath), "best_model.pth")
#         torch.save(state, best_path)
#         print(f"Best model updated: {best_path}")


# def load_checkpoint(model, optimizer=None, filepath=None, device="cuda:0"):
    # """
    # Loads model and optimizer state from a checkpoint.

    # Args:
    #     model: torch.nn.Module
    #     optimizer: torch optimizer (optional)
    #     filepath: path to checkpoint file (if None, auto-loads best_model.pth)
    #     device: device to load checkpoint to
    # Returns:
    #     start_epoch (int): next epoch to continue from
    # """
    # if filepath is None:
    #     filepath = os.path.join("./checkpoints", "best_model.pth")

    # if not os.path.exists(filepath):
    #     raise FileNotFoundError(f"Checkpoint not found!!!: {filepath}")

    # checkpoint = torch.load(filepath, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # if optimizer is not None and "optimizer_state_dict" in checkpoint:
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # start_epoch = checkpoint.get("epoch", 0) + 1
    # print(f"Loaded checkpoint '{filepath}' (epoch {start_epoch})")
    # return start_epoch

# def save_checkpoint(model, optimizer, epoch, path, best=False):
#     checkpoint = {
#         "epoch": epoch,
#         "model_state": model.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "best": best,
#     }
#     torch.save(checkpoint, path)
#     if best:
#         print(f"Saved best checkpoint at: {path}")

# def load_checkpoint(model, optimizer):
#     if os.path.exists(LAST_MODEL_PATH):
#         checkpoint = torch.load(LAST_MODEL_PATH)
#         model.load_state_dict(checkpoint["model_state"])
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
#         print(f"Resumed training from epoch {checkpoint['epoch']+1}")
#         return checkpoint["epoch"] + 1  # Resume from next epoch
#     return 1  # Start from first epoch

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # MUST match training setup exactly
    model = SegmentationModel(
        arch="deeplabv3plus",
        encoder="mit_b5",
        weights=None,      # or "imagenet" if you trained with pretrained weights
        num_classes=19
    ).to(device)

    # Your checkpoint uses "model_state_dict"
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights from 'model_state_dict'")
        return model

    # If something unexpected happens
    raise KeyError(
        f"Could not find model weights. Found keys: {checkpoint.keys()}"
    )


'''Prediction UTILS'''
def predict_and_save_masks(model, test_loader, output_dir=OUTPUT_DIR, device=DEVICE):
    """Generates predictions and saves them as image files"""
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, image_ids in tqdm.tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert logits to class indices
            preds = torch.argmax(outputs, dim=1).cpu()
            
            for pred, image_id in zip(preds, image_ids):
                # Convert prediction to PIL image
                pred_image = to_pil_image(pred.byte())
                
                # Save the predicted mask
                pred_image.save(os.path.join(output_dir, f"{image_id}.png"))

    print(f"\nPredictions saved to {output_dir}")

def jsons2jsonl(jsons_dir, jsonl_path):
    """Combine multiple json files to a jsonl ('json lines') file."""
    with open(jsonl_path, 'w') as f:
        for json_path in sorted(Path(jsons_dir).glob("*.json")):
            with open(json_path, "r") as ff:
                f.write(ff.read() + "\n")

# def predict_and_format(model, dataloader, device=DEVICE):
#     """Run inference on the test set and format predictions."""
#     predictions = []
#     with torch.no_grad():
#         for images in tqdm.tqdm(dataloader, desc="Predicting"):
#             images = images.to(device)
#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#             for image_id, pred in zip(images, preds):
#                 # Flatten the prediction and convert to list
#                 pred_list = pred.flatten().tolist()
#                 # Create a dictionary for the current prediction
#                 prediction_entry = {
#                     "image_id": image_id,
#                     "prediction": pred_list
#                 }
#                 predictions.append(prediction_entry)
#     return predictions

# def save_predictions(predictions, output_jsonl):
#     """Save predictions to a JSONL file."""
#     with open(output_jsonl, 'w') as f:
#         for entry in predictions:
#             if isinstance(entry["prediction"], torch.Tensor): 
#                 entry["prediction"] = entry["prediction"].tolist()
#             else: entry["prediction"]
            
#             json.dump(entry, f)
#             f.write('\n')

# def compress_submission(output_jsonl, output_zip):
#     """Compress the JSONL file into a zip archive."""
#     with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         zipf.write(output_jsonl, os.path.basename(output_jsonl))

def mask_to_polygons(mask):
    """
    Converts a binary mask to a list of polygon points.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if len(contour) >= 3:  # Only consider contours with at least 3 points
            polygon = [[float(point[0][0]), float(point[0][1])] for point in contour]
            polygons.append(polygon)
    
    return polygons

def predict_and_format(model, dataloader, device="cuda:0"):
    """Run inference and format results according to challenge specifications."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, image_ids, image_sizes in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Convert to class indices

            for i, image_id in enumerate(image_ids):
                image_width, image_height = image_sizes[i]  # Tuple: (width, height)
                pred_mask = preds[i]

                unique_classes = np.unique(pred_mask)
                shapes = []

                for class_id in unique_classes:
                    if class_id == 0:  # Assuming 0 is background
                        continue

                    binary_mask = (pred_mask == class_id).astype(np.uint8)
                    polygons = mask_to_polygons(binary_mask)

                    if polygons:
                        shapes.append({
                            "points": polygons,
                            "shape_type": "polygon",
                            "label": get_class_mapping[class_id, "Unknown"]  # Convert class ID to label
                        })

                prediction_entry = {
                    "imageName": image_id,
                    "imageWidth": int(image_width),
                    "imageHeight": int(image_height),
                    "shapes": shapes
                }
                predictions.append(prediction_entry)

    return predictions

def save_predictions(predictions, output_jsonl):
    """Saves predictions in JSONL format."""
    with open(output_jsonl, 'w') as f:
        for entry in predictions:
            json.dump(entry, f)
            f.write('\n')  # JSONL format: one JSON object per line

# Function to convert model predictions to JSONL format
def process_predictions(image_name, image_width, image_height, prediction, class_mapping, threshold=0.5):
    """
    Converts model prediction to the required JSONL format.
    - image_name: Filename of the image
    - image_width, image_height: Dimensions of the image
    - prediction: Model output (logits)
    - class_mapping: Dictionary mapping class indices to labels
    - threshold: Threshold to convert logits to binary mask
    """
    pred_mask = torch.sigmoid(prediction) > threshold  # Convert logits to binary mask
    pred_mask = pred_mask.squeeze(0).cpu().numpy()  # Remove batch dim and convert to NumPy

    shapes = []
    for class_idx in range(1, pred_mask.shape[0]):  # Skip background (index 0)
        mask = pred_mask[class_idx]
        label = class_mapping.get(class_idx, "Unknown")
        
        # Extract polygon points (replace this with a better contour detection if needed)
        points = list(zip(*mask.nonzero()))  # Get (x, y) coordinates
        points = [[float(x), float(y)] for x, y in points]

        if points:
            shapes.append({
                "points": points,
                "shape_type": "polygon",
                "label": label
            })

    return {
        "imageName": image_name,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "shapes": shapes
    }

def predict_and_convert_to_jsonl(model, test_dataset_path, output_jsonl_path, device):
    """Makes predictions on the test dataset and saves them to a JSONL file."""

    transform = T.Compose([
        T.ToTensor(),
    ])

    results = []

    # Assuming test_dataset_path contains image files directly
    image_files = [f for f in os.listdir(test_dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(test_dataset_path, image_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(input_tensor) # Replace with your model's prediction logic.

        shapes = process_prediction(prediction, width, height)

        result = {
            "imageName": image_file,
            "imageWidth": width,
            "imageHeight": height,
            "shapes": shapes,
        }
        results.append(result)

    # Save results to JSONL
    with open(output_jsonl_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')


def process_prediction(prediction, width, height):
    """Processes the model's prediction to extract polygons and labels."""

    # Assuming prediction is a segmentation mask (e.g., from a semantic segmentation model)
    # Adjust this based on your model's output format.

    # Example: Assuming prediction has shape (1, num_classes, H, W)
    mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Example label mapping (replace with your actual labels)
    label_map = {
        1: "Spalling",
        2: "PEquipment",
        # Add more labels as needed
    }

    shapes = []
    unique_labels = np.unique(mask)

    for label_value in unique_labels:
        if label_value == 0:  # Skip background
            continue

        label_mask = (mask == label_value).astype(np.uint8) * 255

        # Find contours using OpenCV
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Simplify the contour (optional, but can reduce the number of points)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert contour points to the desired format
            points = [[float(x), float(y)] for x, y in approx.squeeze()]

            if len(points) >= 3: # Ensure at least 3 points for a polygon.
                shapes.append({
                    "points": points,
                    "shape_type": "polygon",
                    "label": label_map.get(label_value, "Unknown"), #Get label, or unknown if not in map
                })

    return shapes


def visualize_mask(mask_tensor):
    """Convert predicted mask tensor to an RGB image for visualization."""
    # Shape: (C, H, W) → (H, W, C)
    mask_np = mask_tensor.cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8)
    mask_np = np.moveaxis(mask_np, 0, -1)
    mask_color = np.zeros((*mask_np.shape[:2], 3), dtype=np.uint8)
    
    # Generate consistent color map
    np.random.seed(42)  # Set seed for consistent colors across runs
    COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
    
    for c in range(NUM_CLASSES):
        mask_color[mask_np[..., c] == 1] = COLORS[c]

    # Add class labels
    labeled_mask = mask_color.copy()
    for c in range(NUM_CLASSES):
        if mask_np[..., c].sum() > 0:
            # Find the center of the mask for that class
            y, x = np.where(mask_np[..., c] == 1)
            if len(x) > 0 and len(y) > 0:
                center_x, center_y = int(np.mean(x)), int(np.mean(y))

                label_text = CLASS_LABELS[c]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                cv2.putText(
                    labeled_mask,
                    label_text,
                    (center_x, center_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=(0, 0, 0),  # Black border
                    thickness=thickness + 2,
                    lineType=cv2.LINE_AA
                )
            
                cv2.putText(
                    labeled_mask,
                    label_text,
                    (center_x, center_y),
                    fontFace=font,
                    fontScale=font_scale,
                    color=tuple(int(c) for c in COLORS[c]),  # Class color
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
    return labeled_mask


# Function to compute pos_weight for each class
def compute_class_pos_weights(dataset, num_classes):
    loader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=False)  # Small batch
    pos_counts = torch.zeros(num_classes, dtype=torch.float64)  # To hold positive counts for each class
    neg_counts = torch.zeros(num_classes, dtype=torch.float64)  # To hold negative counts for each class

    for images, masks in tqdm(loader, desc="Computing pos_weight"):
        # Flatten the masks (H * W * batch_size) -> (batch_size * H * W, num_classes)
        masks = masks.view(-1, num_classes)  # (batch_size * H * W, num_classes)

        # Count positives and negatives for each class
        pos_counts += masks.sum(dim=0)  # Sum across pixels in the batch for positives (1s)
        neg_counts += (1 - masks).sum(dim=0)  # Sum across pixels in the batch for negatives (0s)

    # Compute pos_weight (negative/positive ratio for each class)
    pos_weight = neg_counts / (pos_counts + 1e-6)  # Adding small value to avoid division by zero

    # Convert to tensor of shape (1, 19, 1, 1)
    pos_weight = pos_weight.to(dtype=torch.float32).view(1, num_classes, 1, 1)

    return pos_weight