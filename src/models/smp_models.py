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
        
        # segmentation model pytorch
        # deeplabv3, unet, unetplusplus, fpn, linknet, manet, pspnet, pan, upernet
        try:
            model_fn = getattr(smp, architecture)
            model = model_fn(**model_params)
        except AttributeError:
            raise ValueError(f"Unsupported architecture: {architecture}. Please check if the model exists in segmentation_models_pytorch")
            
        return model

    def forward(self, x):
        output = self.model(x)
        return {'out': output}  # Match torchvision format