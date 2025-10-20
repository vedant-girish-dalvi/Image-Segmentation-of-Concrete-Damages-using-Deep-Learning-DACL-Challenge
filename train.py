import os
import yaml
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import JaccardIndex
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from model import SegmentationModel
from Dacl_Dataset import train_dataset, validation_dataset
from utils import save_checkpoint, load_checkpoint
from torch.amp import GradScaler, autocast

# -----------------------
# Load YAML configuration
# -----------------------
with open("config_training_rtx4500.yaml", "r") as file:
    cfg = yaml.safe_load(file)

# Set random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ["WANDB_SILENT"] = "true"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load config parameters
# -----------------------
ARCHITECTURE = cfg["model"]["architecture"]
ENCODER = cfg["model"]["encoder"]
WEIGHTS = cfg["model"]["weights"]
NUM_CLASSES = cfg["model"]["num_classes"]

EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]
NUM_WORKERS = cfg["training"]["num_workers"]
PIN_MEMORY = cfg["training"]["pin_memory"]
SHUFFLE = cfg["training"]["shuffle"]
LOAD_MODEL = cfg["training"]["load_model"]

OPT_NAME = cfg["optimizer"]["name"]
OPT_PARAMS = cfg["optimizer"]["params"]

SCHED_NAME = cfg["scheduler"]["name"]
SCHED_PARAMS = cfg["scheduler"]["params"]

AMP_ENABLED = cfg["amp"]["enabled"]
CKPT_FOLDER = cfg.get("checkpoints", {}).get("folder", "./checkpoints")

os.makedirs(CKPT_FOLDER, exist_ok=True)

# -----------------------
# Training function
# -----------------------
def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs, device="cuda:0"):
    scaler = GradScaler(enabled=AMP_ENABLED)

    if LOAD_MODEL:
        print("Resuming training from saved checkpoint.....")
        start_epoch = load_checkpoint(model, optimizer)
    else:
        start_epoch = 0

    model.to(device)
    best_val_loss = float("inf")

    print(f"---------------Training {ARCHITECTURE} ({ENCODER}) on {device}-------------------\n")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_metric = JaccardIndex("multilabel", num_labels=NUM_CLASSES, average=None).to(device)
        val_metric = JaccardIndex("multilabel", num_labels=NUM_CLASSES, average=None).to(device)
        train_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast(enabled=AMP_ENABLED, device_type='cuda'):
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                masks = masks.permute(0, 3, 1, 2).float()
                loss = loss_function(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_metric.update((preds > 0.5).int(), masks.int())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_metric.compute().mean().item()
        train_metric.reset()

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_train_loss)
        else:
            scheduler.step()

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                with autocast(enabled=AMP_ENABLED, device_type='cuda'):
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)
                    masks = masks.permute(0, 3, 1, 2).float()
                    loss = loss_function(outputs, masks)
                val_loss += loss.item()
                val_metric.update((preds > 0.5).int(), masks.int())

            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_metric.compute().mean().item()
            val_metric.reset()

        print(f"\nEpoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train mIoU: {avg_train_iou:.4f} | Val mIoU: {avg_val_iou:.4f}\n")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_mIoU": avg_train_iou,
            "val_mIoU": avg_val_iou
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(CKPT_FOLDER, f"{ARCHITECTURE}_{ENCODER}_epoch{epoch+1}_val_loss{best_val_loss}.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path, best=True)

            # ---- Upload to W&B ----
            # artifact = wandb.Artifact(
            #     name=f"best_model_{ARCHITECTURE}_{ENCODER}",
            #     type="model",
            #     description=f"Best checkpoint at epoch {epoch+1} (val_loss={avg_val_loss:.4f})",
            # )
            # artifact.add_file(ckpt_path)
            # wandb.log_artifact(artifact)
            # print(f" Uploaded best model to W&B as artifact: best_model_{ARCHITECTURE}_{ENCODER}\n")

        print("-" * 120)

    print(" Training Complete!")


# -----------------------
# Main training loop
# -----------------------
def main():

    # -----------------------
    # Initialize W&B
    # -----------------------
    wandb.login()
    run = wandb.init(
        project="semantic-segmentation",
        config=cfg,
        name=f"{ARCHITECTURE}_{ENCODER}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    )

    #Dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=SHUFFLE)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    #Model
    model = SegmentationModel(arch=ARCHITECTURE, encoder=ENCODER, weights=WEIGHTS, num_classes=NUM_CLASSES).to(device=DEVICE)

    # Load and normalize pos_weights
    pos_weights = torch.load("pos_weights.pt").to(dtype=torch.float32, device=DEVICE)
    target_mean = 2.0
    pos_weights = pos_weights * (target_mean / pos_weights.mean())

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="mean")

    # ---------------- Optimizer ----------------
    OPT_PARAMS["lr"] = float(OPT_PARAMS["lr"])
    if "weight_decay" in OPT_PARAMS:
        OPT_PARAMS["weight_decay"] = float(OPT_PARAMS["weight_decay"])
    if OPT_NAME.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **OPT_PARAMS)
    elif OPT_NAME.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **OPT_PARAMS)
    elif OPT_NAME.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **OPT_PARAMS)
    else:
        raise ValueError(f"Unsupported optimizer: {OPT_NAME}")

    # ---------------- Scheduler ----------------
    if SCHED_NAME == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **SCHED_PARAMS)
    elif SCHED_NAME == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(optimizer, **SCHED_PARAMS)
    elif SCHED_NAME == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **SCHED_PARAMS)
    else:
        raise ValueError(f"Unsupported scheduler: {SCHED_NAME}")

    train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, num_epochs=EPOCHS, device=DEVICE)

    wandb.finish()

    
if __name__ == "__main__":
    main()
