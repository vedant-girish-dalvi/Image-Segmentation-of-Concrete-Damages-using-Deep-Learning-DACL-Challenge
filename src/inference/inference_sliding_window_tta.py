import os
import cv2
import json
import random
import torch
import numpy as np
from torchvision import transforms
from utils import load_model, CLASS_LABELS

# ------------------- CONFIG -------------------
IMAGE_DIR = './images/validation'
ANNOTATION_DIR = './annotations/validation'
CHECKPOINT_PATH = './checkpoints/DeepLabV3Plus_mit_b5_e062_miou0.3802_vloss0.1850.pth'

NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiling / inference options
TILE_SIZE = 1024           # tile height/width (square tiles). Pick based on GPU memory.
OVERLAP = 0.25             # fraction overlap between tiles (0.0-0.5 typical)
USE_TTA = True             # apply simple flip TTA
TTA_MODES = ['none', 'hflip', 'vflip', 'hvflip']  # flip modes for TTA

# Thresholds: either a single value or a list/array per class
DEFAULT_THRESHOLD = 0.5
# Example: CLASS_THRESHOLDS = [0.5]*NUM_CLASSES  OR calibrate per-class thresholds list
CLASS_THRESHOLDS = [DEFAULT_THRESHOLD] * NUM_CLASSES

OUTPUT_DIR = "./inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep your color palette
label_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (0, 255, 128), (128, 255, 0), (255, 0, 128),
    (128, 128, 0), (0, 128, 128), (128, 0, 128), (192, 192, 192),
    (128, 128, 128), (64, 64, 64), (255, 255, 255)
]

# ------------------- HELPERS -------------------
def read_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_json_mask(json_path, image_shape):
    """Make multilabel binary mask (C, H, W) from JSON polygons & CLASS_LABELS."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    height, width = image_shape[:2]
    mask = np.zeros((NUM_CLASSES, height, width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in CLASS_LABELS:
            continue
        class_id = CLASS_LABELS.index(label)
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask[class_id], [pts], 1)

    return mask

def overlay_prediction(pred_mask, image, alpha=0.5, text_scale=0.5):
    """Overlay predicted label regions on image (image in RGB)."""
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for c in range(NUM_CLASSES):
        bin_mask = pred_mask[c].astype(np.uint8)
        if bin_mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = label_colors[c]
        for cnt in contours:
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, CLASS_LABELS[c], (cx, cy), font, text_scale, (0,0,0), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def visualize_overlap(pred_mask, gt_mask):
    """Return RGB canvas showing TP (green), FP (red) and TN (dark blue)."""
    h, w = pred_mask.shape[1:3]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(NUM_CLASSES):
        p = pred_mask[c].astype(bool)
        g = gt_mask[c].astype(bool)

        canvas[(p==1) & (g==1)] = [0,255,0]   # TP = green
        canvas[(p==1) & (g==0)] = [0,0,255]   # FP = red
        # leave other pixels black; if you want TN coloring, add here
    return canvas

# ------------------- METRICS -------------------
def compute_metrics(pred_binary, gt_binary):
    """pred_binary and gt_binary shape: (C, H, W), dtype=0/1"""
    eps = 1e-6
    ious = []
    dices = []
    precisions = []
    recalls = []
    f1s = []

    for c in range(NUM_CLASSES):
        p = pred_binary[c].astype(bool)
        g = gt_binary[c].astype(bool)

        TP = np.logical_and(p, g).sum()
        FP = np.logical_and(p, ~g).sum()
        FN = np.logical_and(~p, g).sum()
        # TN not used except pixel accuracy
        # TN = np.logical_and(~p, ~g).sum()

        iou = TP / (TP + FP + FN + eps)
        dice = 2*TP / (2*TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        "IoU_per_class": ious,
        "Dice_per_class": dices,
        "Precision_per_class": precisions,
        "Recall_per_class": recalls,
        "F1_per_class": f1s,
        "IoU_mean": float(np.mean(ious)),
        "Dice_mean": float(np.mean(dices)),
        "Precision_mean": float(np.mean(precisions)),
        "Recall_mean": float(np.mean(recalls)),
        "F1_mean": float(np.mean(f1s)),
        "PixelAcc": float((pred_binary == gt_binary).mean())
    }
    return metrics

# ------------------- INFERENCE: TTA + TILE -------------------
def apply_tta_and_predict(model, patch):
    """
    patch: numpy HxWx3 RGB image (uint8)
    Returns: probabilities numpy array (C, H, W) in float32
    """
    model.eval()
    tensor_transform = transforms.ToTensor()

    preds = []
    for mode in (TTA_MODES if USE_TTA else ['none']):
        if mode == 'none':
            img_mod = patch
        elif mode == 'hflip':
            img_mod = np.flip(patch, axis=1).copy()
        elif mode == 'vflip':
            img_mod = np.flip(patch, axis=0).copy()
        elif mode == 'hvflip':
            img_mod = np.flip(np.flip(patch, axis=0), axis=1).copy()
        else:
            img_mod = patch

        inp = tensor_transform(img_mod).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)
            probs = torch.sigmoid(out).squeeze(0).cpu().numpy()  # (C, H, W)

        # undo transforms on probs
        if mode == 'hflip':
            probs = np.flip(probs, axis=2)
        elif mode == 'vflip':
            probs = np.flip(probs, axis=1)
        elif mode == 'hvflip':
            probs = np.flip(np.flip(probs, axis=1), axis=2)

        preds.append(probs)

    avg = np.mean(preds, axis=0)
    return avg  # float32 numpy (C, H, W)

def sliding_window_inference(image, model, tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    image: HxWx3 RGB numpy
    returns: prob_map (C, H, W) with averaged probabilities
    """
    H, W, _ = image.shape
    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError("Invalid stride: decrease overlap or increase tile_size")

    # accumulators
    prob_map = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = y
            x1 = x
            y2 = min(y1 + tile_size, H)
            x2 = min(x1 + tile_size, W)

            patch = image[y1:y2, x1:x2]

            ph = y2 - y1
            pw = x2 - x1

            # pad to tile_size if needed
            pad_h = tile_size - ph
            pad_w = tile_size - pw
            if pad_h > 0 or pad_w > 0:
                patch_padded = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            else:
                patch_padded = patch

            probs_padded = apply_tta_and_predict(model, patch_padded)  # (C, tile_size, tile_size)
            probs = probs_padded[:, :ph, :pw]  # crop back to original patch size

            prob_map[:, y1:y2, x1:x2] += probs
            count_map[y1:y2, x1:x2] += 1.0

    # avoid division by zero for pixels that were never covered (shouldn't happen)
    count_map[count_map == 0] = 1.0
    prob_map = prob_map / count_map[np.newaxis, :, :]

    return prob_map  # float32

# ------------------- MAIN DRIVER -------------------
def infer_on_image(model, img_path, json_path=None, save=True, verbose=True):
    original_img = read_image_rgb(img_path)
    H, W, _ = original_img.shape

    # Run tiled inference
    prob_map = sliding_window_inference(original_img, model, tile_size=TILE_SIZE, overlap=OVERLAP)

    # Binarize with class thresholds
    thresholds = np.array(CLASS_THRESHOLDS, dtype=np.float32)
    if thresholds.shape[0] != NUM_CLASSES:
        thresholds = np.array([DEFAULT_THRESHOLD]*NUM_CLASSES)
    pred_binary = (prob_map >= thresholds[:, None, None]).astype(np.uint8)

    # Load GT if available & compute metrics
    metrics = None
    if json_path and os.path.exists(json_path):
        gt_mask = parse_json_mask(json_path, original_img.shape)  # (C,H,W) uint8
        # ensure shapes match
        if gt_mask.shape[1:] != pred_binary.shape[1:]:
            # resize GT (nearest) to match prediction resolution (shouldn't be needed)
            resized = np.zeros_like(pred_binary)
            for c in range(NUM_CLASSES):
                resized[c] = cv2.resize(gt_mask[c].astype(np.uint8),
                                        (W, H), interpolation=cv2.INTER_NEAREST)
            gt_mask = resized
        metrics = compute_metrics(pred_binary, gt_mask)

    # Visualizations
    overlay = overlay_prediction(pred_binary, original_img)
    tp_fp_vis = None
    if json_path and os.path.exists(json_path):
        tp_fp_vis = visualize_overlap(pred_binary, gt_mask)

    if save:
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if tp_fp_vis is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_tp_fp.png"), tp_fp_vis)
        # optionally save raw probability maps (npz)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{base}_probs.npz"), probs=prob_map)

    if verbose:
        print(f"Inference finished for {img_path}")
        if metrics:
            print("== METRICS ==")
            print(f"Mean IoU: {metrics['IoU_mean']:.4f}  Mean Dice: {metrics['Dice_mean']:.4f}")
            print(f"Mean Precision: {metrics['Precision_mean']:.4f}  Mean Recall: {metrics['Recall_mean']:.4f}")
            print(f"Mean F1: {metrics['F1_mean']:.4f}  PixelAcc: {metrics['PixelAcc']:.4f}")

    return {"prob_map": prob_map, "pred_mask": pred_binary, "metrics": metrics, "overlay": overlay, "tp_fp_vis": tp_fp_vis}

# ------------------- RUN A RANDOM IMAGE -------------------
def run_inference_random():
    model = load_model(CHECKPOINT_PATH, device=DEVICE)
    model.to(DEVICE)
    model.eval()

    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not images:
        raise RuntimeError("No images found in IMAGE_DIR")

    img_name = random.choice(images)
    img_path = os.path.join(IMAGE_DIR, img_name)
    json_path = os.path.join(ANNOTATION_DIR, os.path.splitext(img_name)[0] + ".json")

    print("Selected:", img_name)
    infer_on_image(model, img_path, json_path, save=True, verbose=True)

if __name__ == "__main__":
    run_inference_random()
