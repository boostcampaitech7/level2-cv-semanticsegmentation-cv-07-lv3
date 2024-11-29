from .torchvision_models import TorchvisionSegmentationModel
from .smp_models import SMPSegmentationModel

def get_model(cfg):
    model_type = cfg['MODEL'].get('TYPE', 'torchvision')
    
    if model_type == 'torchvision':
        return TorchvisionSegmentationModel(cfg)
    elif model_type == 'smp':
        return SMPSegmentationModel(cfg)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")