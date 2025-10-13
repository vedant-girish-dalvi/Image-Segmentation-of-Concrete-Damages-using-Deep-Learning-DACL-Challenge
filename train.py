import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SegmentationModel
# from dataset import train_dataset, validation_dataset
from Dacl_Dataset import train_dataset, validation_dataset
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex
import wandb
from utils import CLASS_LABELS, save_checkpoint, load_checkpoint, compute_class_pos_weights

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ["WANDB_SILENT"] = "true"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_dir = "./dacl_challenge/images"
json_dir = "./dacl_challenge/annotations/train"

# hyperparameters
NUM_CLASSES = 19
EPOCHS = 400        
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_WORKERS=0
PIN_MEMORY=True
LOAD_MODEL = False
SHUFFLE = True
ARCHITECTURE = 'Unet'
ENCODER = 'resnet34'
WEIGHTS = 'imagenet'

wandb.login()

run = wandb.init(
    project="semantic-segmentation", 
    config={                        
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size":BATCH_SIZE,
    },
)

def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs, device="cuda:0"):
    if LOAD_MODEL:
        print("Resuming training from saved checkpoint.....")
        start_epoch = load_checkpoint(model, optimizer)
    else:
        start_epoch = 0
    model.to(device)
    best_val_loss = float('inf')

    print(f"---------------Model training on {device}-------------------\n")

    for epoch in range(start_epoch, num_epochs+1): 
        model.train() 
        train_metric = JaccardIndex("multilabel", num_labels=NUM_CLASSES, average=None).to(device)  
        val_metric = JaccardIndex("multilabel",  num_labels=NUM_CLASSES, average=None).to(device) 
        train_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()   
            outputs = model(images) 
            preds = torch.sigmoid(outputs)
            # print(preds.shape)
            # print(preds)

            # masks_one_hot_temp = F.one_hot(masks, num_classes=NUM_CLASSES).squeeze(1)
            masks = masks.permute(0, 3, 1, 2).float()
            # print(f"\n Output shape and type:{outputs.shape}, {outputs.dtype}, Mask shape: {masks.shape}, {masks.dtype}\n")
            loss = loss_function(outputs, masks)  
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item()
            train_metric.update((preds>0.5).int(), masks.int()) 
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        avg_train_ious = train_metric.compute()
        train_metric.reset()
        avg_train_iou = avg_train_ious.mean().item()

        model.eval() 
        val_loss = 0
       
        with torch.no_grad():  
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images) 
                preds = torch.sigmoid(outputs)
                # print(preds)

                # masks_one_hot_temp = F.one_hot(masks, num_classes=NUM_CLASSES).squeeze(1)
                # masks_one_hot = masks_one_hot_temp.permute(0, 3, 1, 2).float()
                masks = masks.permute(0, 3, 1, 2).float()
                loss = loss_function(outputs, masks)  
                val_loss += loss.item()

                # print(preds.shape(), masks_one_hot.shape())
                val_metric.update((preds>0.5).int(), masks.int())

            avg_val_loss = val_loss / len(val_loader)
            avg_val_ious = val_metric.compute()
            val_metric.reset()
            avg_val_iou = avg_val_ious.mean().item()  
           
    #    # WandB log: Bar Chart
    #     wandb.log({
    #                 f"Validation IoU per class Bar (Epoch {epoch+1})": wandb.plot.bar(
    #         wandb.Table(data=[[CLASS_LABELS[i], avg_val_ious[i].item()] for i in range(NUM_CLASSES)], columns=["Class", "IoU"]
    #                 ),
    #         "Class",
    #         "IoU",
    #         title=f"Validation IoU per Class (Epoch {epoch+1})")
    #             })

        print(f"\n Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train mIOU: {avg_train_iou:.4f}, Val Loss: {avg_val_loss:.4f}, Val mIOU: {avg_val_iou:.4f}\n")
        wandb.log({
            "\n Training loss": avg_train_loss, "Validation loss": avg_val_loss, 
            "\n Train mIOU": avg_train_iou, "Val mIOU": avg_val_iou
                })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, best=True)
            print(f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}\n")
        
        print("*" * 120)

    print("Training Complete!")


def main():

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=SHUFFLE)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    model = SegmentationModel(arch=ARCHITECTURE, encoder=ENCODER, weights=WEIGHTS, num_classes=NUM_CLASSES).to(device=DEVICE)
    
    # pos_weights = compute_class_pos_weights(train_dataset, NUM_CLASSES).to(DEVICE)
    # print(f"pos_weight computed: {pos_weights}")
    # torch.save(pos_weights, "pos_weights.pt")
    pos_weights = torch.load("pos_weights.pt").to(dtype=torch.float32, device=DEVICE)
    # Keep pos_weight values within a safe range
    # pos_weights = torch.clamp(pos_weights, min=1.0, max=5.0)

    # Optional: Normalize to a specific mean if preferred
    target_mean = 2.0
    pos_weights = pos_weights * (target_mean / pos_weights.mean())
    # print(f"pos_weight computed: {pos_weights}\n")
    # print(f"pos_weights shape: {pos_weights.shape}, dtype: {pos_weights.dtype}, device: {pos_weights.device}")
   
    # loss = smp.losses.DiceLoss(mode='multiclass')
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="mean")

    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    train_model(model, train_dataloader, validation_dataloader, loss, optimizer, scheduler, num_epochs=EPOCHS)
    
if __name__ == '__main__':
    main()