import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SMPSegmentationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = self._build_model()

    def _build_model(self):
        model_params = {
            'encoder_name': self.cfg['MODEL'].get('ENCODER', 'resnet50'),
            'encoder_weights': 'imagenet' if self.cfg['MODEL']['PRETRAINED'] else None,
            'in_channels': 3,
            'classes': len(self.cfg['CLASSES'])
        }

        architecture = self.cfg['MODEL'].get('ARCHITECTURE', 'Unet')
        
        if architecture == 'Unet':
            model = smp.Unet(**model_params)
        elif architecture == 'DeepLabV3':
            model = smp.DeepLabV3(**model_params)
        elif architecture == 'DeepLabV3Plus':
            model = smp.DeepLabV3Plus(**model_params)
        elif architecture == 'FPN':
            model = smp.FPN(**model_params)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
            
        return model

    def forward(self, x):
        output = self.model(x)
        return {'out': output}  # Match torchvision format