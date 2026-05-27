import os
import cv2
import random
import torch
import numpy as np
from torchvision import transforms
from utils import load_model, CLASS_LABELS

# ------------------- CONFIG -------------------
TEST_IMAGE_DIR = './our_dataset'
CHECKPOINT_PATH = './checkpoints/DeepLabV3Plus_mit_b5_e062_miou0.3802_vloss0.1850.pth'

NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE_SIZE = 1024
OVERLAP = 0.25
USE_TTA = True
TTA_MODES = ['none', 'hflip', 'vflip', 'hvflip']

DEFAULT_THRESHOLD = 0.5
CLASS_THRESHOLDS = [DEFAULT_THRESHOLD] * NUM_CLASSES

OUTPUT_DIR = "./test_inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette (your CLASS_COLORS converted to list order)
label_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 165, 0), (128, 0, 128), (0, 255, 255), (255, 192, 203),
    (139, 69, 19), (128, 128, 128), (0, 128, 128), (50, 205, 50),
    (75, 0, 130), (255, 20, 147), (0, 191, 255), (139, 0, 139),
    (173, 255, 47), (220, 20, 60), (0, 100, 0)
]

# ------------------- HELPERS -------------------
def read_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def overlay_prediction(pred_mask, image, alpha=0.5, text_scale=0.5):
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

    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# ------------------- TTA + TILE -------------------
def apply_tta_and_predict(model, patch):
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
            probs = torch.sigmoid(out).squeeze(0).cpu().numpy()

        if mode == 'hflip':
            probs = np.flip(probs, axis=2)
        elif mode == 'vflip':
            probs = np.flip(probs, axis=1)
        elif mode == 'hvflip':
            probs = np.flip(np.flip(probs, axis=1), axis=2)

        preds.append(probs)

    return np.mean(preds, axis=0)

def sliding_window_inference(image, model, tile_size=TILE_SIZE, overlap=OVERLAP):
    H, W, _ = image.shape
    stride = int(tile_size * (1 - overlap))

    prob_map = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)

            patch = image[y:y2, x:x2]
            ph, pw = patch.shape[:2]

            pad_h = tile_size - ph
            pad_w = tile_size - pw
            if pad_h > 0 or pad_w > 0:
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            probs_full = apply_tta_and_predict(model, patch)
            probs = probs_full[:, :ph, :pw]

            prob_map[:, y:y2, x:x2] += probs
            count_map[y:y2, x:x2] += 1

    count_map[count_map == 0] = 1
    return prob_map / count_map[np.newaxis, :, :]

# ------------------- MAIN INFERENCE -------------------
def infer_test_image(model, img_path, save=True):
    original = read_image_rgb(img_path)
    H, W = original.shape[:2]

    prob_map = sliding_window_inference(original, model)
    pred_binary = (prob_map >= np.array(CLASS_THRESHOLDS)[:, None, None]).astype(np.uint8)

    overlay = overlay_prediction(pred_binary, original)

    if save:
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{base}_probs.npz"), probs=prob_map)

    print(f"[DONE] Saved output for {img_path}")
    return {"prob_map": prob_map, "pred_mask": pred_binary, "overlay": overlay}

# ------------------- RANDOM TEST IMAGE -------------------
def run_inference_random_test():
    model = load_model(CHECKPOINT_PATH, device=DEVICE).to(DEVICE)
    model.eval()

    imgs = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not imgs:
        raise RuntimeError("No test images found!")

    img_name = random.choice(imgs)
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)

    print("Selected test image:", img_name)
    infer_test_image(model, img_path, save=True)

if __name__ == "__main__":
    run_inference_random_test()
