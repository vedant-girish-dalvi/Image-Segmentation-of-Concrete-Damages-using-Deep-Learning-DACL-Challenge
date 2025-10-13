import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):
    def __init__(self, arch, encoder, weights, num_classes):
        super(SegmentationModel, self).__init__()
        self.model = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=num_classes)

    def forward(self, images):
        return self.model(images)
    