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
        
        # torchvision segmentation models
        # fcn : fcn_resnet50, fcn_resnet101
        # deeplabv3 : deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
        try:
            model_fn = getattr(models.segmentation, model_name)
            model = model_fn(pretrained=self.cfg['MODEL']['PRETRAINED'])
        except AttributeError:
            raise ValueError(f"Unsupported model name: {model_name}. Please check if the model exists in torchvision.models.segmentation")
        
        # Modify the classifier based on model architecture
        if model_name.startswith('fcn'):
            model.classifier[4] = nn.Conv2d(512, len(self.cfg['CLASSES']), kernel_size=1)
        elif model_name.startswith('deeplabv3'):
            model.classifier[-1] = nn.Conv2d(256, len(self.cfg['CLASSES']), kernel_size=1)
            
        return model

    def forward(self, x):
        return self.model(x)