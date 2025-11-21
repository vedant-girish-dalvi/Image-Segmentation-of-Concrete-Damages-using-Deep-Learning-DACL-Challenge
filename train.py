import os
import math
import yaml
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from torchmetrics import JaccardIndex

from model import SegmentationModel
from Dacl_Dataset import train_dataset, validation_dataset
from utils import set_seed, save_checkpoint, resume_from_checkpoint
from early_stopping import EarlyStopping

# optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# segmentation_models_pytorch utilities (Dice loss)
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

# -----------------------
# Read config
# -----------------------
with open("config_training_rtx4500.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# shorthand
TRAIN_CFG = cfg.get("training", {})
MODEL_CFG = cfg.get("model", {})
OPT_CFG = cfg.get("optimizer", {})
SCHED_CFG = cfg.get("scheduler", {})
AMP_CFG = cfg.get("amp", {})
CKPT_CFG = cfg.get("checkpoints", {})
EARLY_CFG = cfg.get("early_stopping", {})

# Set seed
SEED = TRAIN_CFG.get("seed", 42)
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Hyperparams
# -----------------------
ARCHITECTURE = MODEL_CFG.get("architecture", "DeepLabV3Plus")
ENCODER = MODEL_CFG.get("encoder", "mit_b5")
WEIGHTS = MODEL_CFG.get("weights", "imagenet")
NUM_CLASSES = MODEL_CFG.get("num_classes", 19)

EPOCHS = int(TRAIN_CFG.get("epochs", 250))
BATCH_SIZE = int(TRAIN_CFG.get("batch_size", 8))
NUM_WORKERS = int(TRAIN_CFG.get("num_workers", 4))
CLIP_GRAD_NORM = float(TRAIN_CFG.get("clip_grad_norm", 1.0))
PIN_MEMORY = bool(TRAIN_CFG.get("pin_memory", True))
SHUFFLE = bool(TRAIN_CFG.get("shuffle", True))
LOAD_MODEL = bool(TRAIN_CFG.get("load_model", False))

OPT_NAME = OPT_CFG.get("name", "AdamW")
OPT_PARAMS = OPT_CFG.get("params", {"lr": 3e-4, "weight_decay": 1e-4})
LR = float(OPT_PARAMS.get("lr", 3e-4))
WEIGHT_DECAY = float(OPT_PARAMS.get("weight_decay", 1e-4))
DIFFERENTIAL_LR = bool(OPT_PARAMS.get("differential_lr", True))
ENCODER_LR_FACTOR = float(OPT_PARAMS.get("encoder_lr_factor", 0.1))

SCHED_NAME = SCHED_CFG.get("name", "WarmupCosine")
SCHED_PARAMS = SCHED_CFG.get("params", {})
WARMUP_EPOCHS = int(SCHED_PARAMS.get("warmup_epochs", 10))
MIN_LR_RATIO = float(SCHED_PARAMS.get("min_lr_ratio", 0.1))

AMP_ENABLED = bool(AMP_CFG.get("enabled", True))
USE_AMP = AMP_ENABLED and DEVICE.type == "cuda"
AUTOCAST_DEVICE = 'cuda' if DEVICE.type == 'cuda' else 'cpu'

CKPT_FOLDER = CKPT_CFG.get("folder", "./checkpoints")
CKPT_PATH = CKPT_CFG.get("filepath", None)
os.makedirs(CKPT_FOLDER, exist_ok=True)

EARLY_ENABLED = bool(EARLY_CFG.get("enabled", True))
EARLY_PATIENCE = int(EARLY_CFG.get("patience", 15))
EARLY_MIN_DELTA = float(EARLY_CFG.get("min_delta", 0.001))
EARLY_ALPHA = float(EARLY_CFG.get("alpha", 0.1))  # val_score = mIoU - alpha * val_loss

WANDB_PROJECT = cfg.get("wandb", {}).get("project", "semantic-segmentation")

# -----------------------
# Helpers
# -----------------------

def build_optimizer(model: nn.Module):
    # split params by name: encoder vs others
    if DIFFERENTIAL_LR:
        encoder_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder" in name or "backbone" in name:
                encoder_params.append(p)
            else:
                other_params.append(p)
        param_groups = [
            {"params": encoder_params, "lr": LR * ENCODER_LR_FACTOR},
            {"params": other_params, "lr": LR},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": LR}]

    if OPT_NAME.lower() == "adamw":
        opt = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    elif OPT_NAME.lower() == "adam":
        opt = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
    elif OPT_NAME.lower() == "sgd":
        opt = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {OPT_NAME}")
    return opt


def warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.1):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            t = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda)


# Combined loss: BCEWithLogits (with pos_weight) + Dice (multilabel)
class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, bce_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        if SMP_AVAILABLE:
            self.dice = smp.losses.DiceLoss(mode="multilabel")
        else:
            self.dice = None
        self.bw = bce_weight
        self.dw = dice_weight

    def forward(self, logits, masks):
        # logits: BxCxHxW, masks: BxCxHxW (float)
        bce_loss = self.bce(logits, masks)
        if self.dice is not None:
            dice_loss = self.dice(torch.sigmoid(logits), masks)
            return self.bw * bce_loss + self.dw * dice_loss
        else:
            return bce_loss


# TTA inference (multi-scale averaging + horizontal flip)
def tta_predict(model, images, scales=(1.0, 0.75, 1.25), device=DEVICE):
    # images: BxCxHxW
    model.eval()
    preds = []
    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                size = (int(images.shape[2] * s), int(images.shape[3] * s))
                img_s = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
            else:
                img_s = images

            out = torch.sigmoid(model(img_s.to(device)))
            out = F.interpolate(out, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False)
            preds.append(out)

            # horizontal flip
            imgs_flip = torch.flip(img_s, dims=[3])
            out_flip = torch.sigmoid(model(imgs_flip.to(device)))
            out_flip = torch.flip(out_flip, dims=[3])
            out_flip = F.interpolate(out_flip, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False)
            preds.append(out_flip)

    preds = torch.stack(preds, dim=0).mean(dim=0)  # average across TTA variations
    return preds


# -----------------------
# Training loop
# -----------------------

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, scheduler, num_epochs: int, device=DEVICE):

    scaler = GradScaler(enabled=AMP_ENABLED)

    start_epoch = 0
    if LOAD_MODEL and CKPT_PATH and os.path.exists(CKPT_PATH):
        print(f"Resuming from checkpoint: {CKPT_PATH}")
        start_epoch = resume_from_checkpoint(
                        model, optimizer, scheduler=scheduler, scaler=scaler, filepath=CKPT_PATH, device=device
        )


    model.to(device)

    best_val_score = -1e9
    early_stopper = EarlyStopping(patience=EARLY_PATIENCE, min_delta=EARLY_MIN_DELTA) if EARLY_ENABLED else None

    # metrics
    jaccard_val = JaccardIndex(task="multilabel", num_labels=NUM_CLASSES, average=None).to(device)

    print(f"---------------------------------- Training {ARCHITECTURE} + {ENCODER} on {DEVICE} ----------------------------------")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()

        # freeze encoder for first warmup epochs
        if epoch < WARMUP_EPOCHS:
            for name, p in model.named_parameters():
                if "encoder" in name or "backbone" in name:
                    p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = True

        running_loss = 0.0
        jaccard_train = JaccardIndex(task="multilabel", num_labels=NUM_CLASSES, average=None).to(device)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Train")
        for step, (images, masks) in pbar:
            images = images.to(device)
            # expect masks as BxHxWxC or BxCxHxW, try to normalize to BxCxHxW float
            if masks.ndim == 4 and masks.shape[1] in (1, NUM_CLASSES):
                # assume BxC,H,W
                masks = masks.float().to(device)
            elif masks.ndim == 4 and masks.shape[-1] in (1, NUM_CLASSES):
                masks = masks.permute(0, 3, 1, 2).float().to(device)
            else:
                masks = masks.float().to(device)

            optimizer.zero_grad()
            with autocast(enabled=USE_AMP, device_type=AUTOCAST_DEVICE):
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()

            # gradient clipping
            if CLIP_GRAD_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).int()
            jaccard_train.update(preds, masks.int())

            if step % 20 == 0:
                pbar.set_postfix({'loss': f"{running_loss / (step+1):.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        class_ious_train = jaccard_train.compute().detach().cpu().numpy()
        train_mIoU = float(class_ious_train.mean())
        jaccard_train.reset()

        # ---------------- validation ----------------
        model.eval()
        val_running_loss = 0.0
        jaccard_val.reset()

        with torch.no_grad():
            for step, (images, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val")):
                images = images.to(device)
                if masks.ndim == 4 and masks.shape[1] in (1, NUM_CLASSES):
                    masks = masks.float().to(device)
                elif masks.ndim == 4 and masks.shape[-1] in (1, NUM_CLASSES):
                    masks = masks.permute(0, 3, 1, 2).float().to(device)
                else:
                    masks = masks.float().to(device)

                if SCHED_NAME == "ReduceLROnPlateau":
                    # standard forward
                    with autocast(enabled=AMP_ENABLED, device_type='cuda'):
                        logits = model(images)
                        loss = loss_fn(logits, masks)
                    preds = torch.sigmoid(logits)
                else:
                    # use TTA prediction for better validation estimate
                    preds = tta_predict(model, images, scales=SCHED_PARAMS.get("tta_scales", [1.0, 0.75, 1.25]), device=device)
                    # compute loss on original forward pass (non-tta) to keep loss comparable
                    with autocast(enabled=USE_AMP, device_type=AUTOCAST_DEVICE):
                        logits = model(images)
                        loss = loss_fn(logits, masks)

                val_running_loss += loss.item()
                jaccard_val.update((preds > 0.5).int(), masks.int())

        avg_val_loss = val_running_loss / len(val_loader)
        class_ious = jaccard_val.compute().detach().cpu().numpy()
        val_mIoU = float(class_ious.mean())
        jaccard_val.reset()

        # scheduler step
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-val_mIoU)  # maximize mIoU as per EarlyStopping Class
        else:
            scheduler.step()

        # Compose combined val score
        val_score = val_mIoU - EARLY_ALPHA * avg_val_loss

        # save best model by val_score
        if val_score > best_val_score:
            best_val_score = val_score
            ckpt_name = f"{ARCHITECTURE}_{ENCODER}_e{epoch+1:03d}_miou{val_mIoU:.4f}_vloss{avg_val_loss:.4f}.pth"
            ckpt_path = os.path.join(CKPT_FOLDER, ckpt_name)
            save_checkpoint(model, optimizer, epoch, ckpt_path, best=True)
            print(f"Saved best model: {ckpt_path}")

        # logging to W&B
        if WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mIoU": train_mIoU,
                "val_mIoU": val_mIoU,
            }
            # log per-class IoUs for val
            for i, iou in enumerate(class_ious):
                log_dict[f"val_class_{i}_IoU"] = float(iou)
            # small visualization of first batch of validation
            try:
                imgs_vis = images.detach().cpu().permute(0,2,3,1).numpy()[:3]
                # if float in [0,1] -> convert
                if imgs_vis.dtype == np.float32 or imgs_vis.max() <= 1.0:
                    imgs_vis = (np.clip(imgs_vis,0,1) * 255).astype(np.uint8)
                else:
                    imgs_vis = imgs_vis.astype(np.uint8)
                preds_vis = (preds.detach().cpu().numpy()[:3] > 0.5).astype(np.uint8)  # BxCxHxW
                # create overlay images for first 3
                vis_list = []
                for b in range(min(3, imgs_vis.shape[0])):
                    img = imgs_vis[b]
                    # pick one class to overlay (example: class 0)
                    mask_b = preds_vis[b][0] * 255
                    overlay = img.copy()
                    overlay[mask_b == 255] = (overlay[mask_b == 255] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                    vis_list.append(wandb.Image(overlay, caption=f"pred_e{epoch+1}"))
                log_dict["val_samples"] = vis_list
            except Exception:
                pass

            # indicate whether early-stopping is active this epoch
            log_dict["early_stopping_active"] = bool(epoch >= 15)
            wandb.log(log_dict)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train mIoU: {train_mIoU:.4f} | Val mIoU: {val_mIoU:.4f}")

        # ---------------- Early Stopping (delayed until after epoch 15) ----------------
        if EARLY_ENABLED and early_stopper is not None:
            if epoch >= 15:
                # activate early stopping checks from epoch 16 onward (0-based epoch >=15)
                early_stopper.step(-val_score)
                if early_stopper.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {early_stopper.patience} epochs).")
                    break
            else:
                # Not checking early stopping yet; print/log for clarity
                print(f"Skipping early stopping check until after epoch 15 (current epoch: {epoch+1})")

    print("Training finished.")


# -----------------------
# Main
# -----------------------

def main():
    # initialize wandb if available
    if WANDB_AVAILABLE:
        wandb.login()
        run = wandb.init(project=WANDB_PROJECT, config=cfg,
                         name=f"{ARCHITECTURE}_{ENCODER}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")

    # data loaders (assumes train_dataset & validation_dataset are pre-built objects)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # build model from your module
    model = SegmentationModel(arch=ARCHITECTURE, encoder=ENCODER, weights=WEIGHTS, num_classes=NUM_CLASSES)

    # load pos_weights if present
    pos_weights_path = Path("pos_weights.pt")
    pos_weight = None
    if pos_weights_path.exists():
        pos_weight = torch.load(pos_weights_path)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)

        # normalize scale to target mean (as you did previously)
        target_mean = float(cfg.get("pos_weight_target_mean", 2.0))
        pos_weight = pos_weight * (target_mean / pos_weight.mean())
    
    # combined loss
    loss_fn = CombinedLoss(pos_weight=pos_weight, bce_weight=0.7, dice_weight=0.3)

    # optimizer & scheduler
    optimizer = build_optimizer(model)
    if SCHED_NAME == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **SCHED_CFG.get("params", {}))
    else:
        scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS, min_lr_ratio=MIN_LR_RATIO)

    # print optimizer groups
    print("Optimizer param groups LRs:", [g.get("lr") for g in optimizer.param_groups])

    train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_epochs=EPOCHS, device=DEVICE)

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
