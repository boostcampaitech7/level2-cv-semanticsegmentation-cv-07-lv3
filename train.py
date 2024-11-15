import os
import random
import argparse
import yaml
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torch.utils.data import DataLoader

from src.models import get_model
from src.dataset import XRayDataset
from src.loss import LossFactory
from src.optimizer import OptimizerFactory
from src.trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, default='torchvision_fcn_resnet50.yaml',
                        help='name of config file in configs directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_name):
    """Load config file from configs directory"""
    config_path = os.path.join('configs', config_name)
    if not os.path.exists(config_path):
        print(f'Config file not found: {config_path}')
        exit(1)
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f'Error loading config file: {e}')
            exit(1)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_transforms(cfg):
    """Get data transforms for train and validation"""
    image_size = cfg['DATASET'].get('IMAGE_SIZE', 512)  # Default to 512 if not specified
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        # Add more augmentations here if needed
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
    ])
    
    return train_transform, val_transform


def main():
    """Main training function"""
    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config)

    wandb.init(
        project = "Segmentation", 
        entity = 'jhs7027-naver', 
        group = 'test', 
        name = cfg['WANDB']['NAME'], 
        config = {
            "IMAGE_SIZE": cfg['DATASET'].get('IMAGE_SIZE'),
            "BATCH_SIZE": cfg['DATASET'].get('BATCH_SIZE'),
            "NUM_WORKERS": cfg['DATASET'].get('NUM_WORKERS'),
            "ENCODER": cfg['MODEL'].get('ENCODER'),
            "NUM_EPOCHS": cfg['TRAIN'].get('NUM_EPOCHS'),
            "VAL_EVERY": cfg['TRAIN'].get('VAL_EVERY'),
            "LEARNING_RATE": cfg['TRAIN'].get('LEARNING_RATE'),
            "WEIGHT_DECAY": cfg['TRAIN'].get('WEIGHT_DECAY'),
            "RANDOM_SEED": cfg['TRAIN'].get('RANDOM_SEED'),
        }
    )
    
    # Set random seed
    set_seed(cfg['TRAIN']['RANDOM_SEED'])
    
    # Create save directory if it doesn't exist
    save_dir = cfg['TRAIN'].get('SAVED_DIR', 'checkpoints')  # Default to 'checkpoints' if not specified
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup transforms
    train_transform, val_transform = get_transforms(cfg)
    
    # Setup datasets
    train_dataset = XRayDataset(
        cfg=cfg,
        is_train=True,
        transforms=train_transform
    )
    
    valid_dataset = XRayDataset(
        cfg=cfg,
        is_train=False,
        transforms=val_transform
    )
    
    # Setup dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg['DATASET']['BATCH_SIZE'],
        shuffle=True,
        num_workers=cfg['DATASET']['NUM_WORKERS'],
        drop_last=True
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg['DATASET']['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,  # Prevent memory issues during validation
        drop_last=False
    )
    
    # Setup model
    model = get_model(cfg)
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup loss function and optimizer using factories
    criterion = LossFactory.get_loss(cfg['LOSS'])
    optimizer = OptimizerFactory.get_optimizer(cfg['OPTIMIZER'], model.parameters())
    
    # Setup trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()