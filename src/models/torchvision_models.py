import torch
import torch.nn as nn
from torchvision import models

class TorchvisionSegmentationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = self._build_model()

    def _build_model(self):
        model_name = self.cfg['MODEL'].get('NAME', 'fcn_resnet50')
        
        if model_name == 'fcn_resnet50':
            model = models.segmentation.fcn_resnet50(
                pretrained=self.cfg['MODEL']['PRETRAINED']
            )
        elif model_name == 'deeplabv3_resnet50':
            model = models.segmentation.deeplabv3_resnet50(
                pretrained=self.cfg['MODEL']['PRETRAINED']
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Modify the classifier for our number of classes
        if model_name.startswith('fcn'):
            model.classifier[4] = nn.Conv2d(512, len(self.cfg['CLASSES']), kernel_size=1)
        elif model_name.startswith('deeplabv3'):
            model.classifier[-1] = nn.Conv2d(256, len(self.cfg['CLASSES']), kernel_size=1)
            
        return model

    def forward(self, x):
        return self.model(x)