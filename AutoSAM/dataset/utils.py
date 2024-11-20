import os
import pickle
from dataset.XRay import XRayDataset
from torch.utils.data import DataLoader
import torch
import albumentations as A

def get_transforms(cfg):
    """Get data transforms for train and validation"""
    image_size = cfg['DATASET'].get('IMAGE_SIZE', 512)
    aug_cfg = cfg.get('AUGMENTATION', {})
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        
        # Basic augmentations
        A.HorizontalFlip(
            p=aug_cfg.get('HORIZONTAL_FLIP', {}).get('P', 0.5)
        ) if aug_cfg.get('HORIZONTAL_FLIP', {}).get('ENABLED', True) else A.NoOp(),
        
        A.VerticalFlip(
            p=aug_cfg.get('VERTICAL_FLIP', {}).get('P', 0.5)
        ) if aug_cfg.get('VERTICAL_FLIP', {}).get('ENABLED', True) else A.NoOp(),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.get('RANDOM_BRIGHTNESS_CONTRAST', {}).get('BRIGHTNESS_LIMIT', 0.2),
            contrast_limit=aug_cfg.get('RANDOM_BRIGHTNESS_CONTRAST', {}).get('CONTRAST_LIMIT', 0.2),
            p=aug_cfg.get('RANDOM_BRIGHTNESS_CONTRAST', {}).get('P', 0.5)
        ) if aug_cfg.get('RANDOM_BRIGHTNESS_CONTRAST', {}).get('ENABLED', True) else A.NoOp(),
        
        # Geometric augmentations
        A.Rotate(
            limit=aug_cfg.get('RANDOM_ROTATE', {}).get('LIMIT', 15),
            p=aug_cfg.get('RANDOM_ROTATE', {}).get('P', 0.5)
        ) if aug_cfg.get('RANDOM_ROTATE', {}).get('ENABLED', True) else A.NoOp(),
        
        # Noise and filtering
        A.GaussNoise(
            var_limit=aug_cfg.get('GAUSSIAN_NOISE', {}).get('VAR_LIMIT', [10.0, 50.0]),
            p=aug_cfg.get('GAUSSIAN_NOISE', {}).get('P', 0.3)
        ) if aug_cfg.get('GAUSSIAN_NOISE', {}).get('ENABLED', True) else A.NoOp(),
        
        # Contrast enhancement
        A.CLAHE(
            clip_limit=aug_cfg.get('CLAHE', {}).get('CLIP_LIMIT', 4.0),
            p=aug_cfg.get('CLAHE', {}).get('P', 0.5)
        ) if aug_cfg.get('CLAHE', {}).get('ENABLED', True) else A.NoOp(),
        
        A.RandomGamma(
            gamma_limit=aug_cfg.get('RANDOM_GAMMA', {}).get('GAMMA_LIMIT', [80, 120]),
            p=aug_cfg.get('RANDOM_GAMMA', {}).get('P', 0.3)
        ) if aug_cfg.get('RANDOM_GAMMA', {}).get('ENABLED', True) else A.NoOp(),
        
        # Elastic and grid distortions
        A.ElasticTransform(
            alpha=aug_cfg.get('ELASTIC_TRANSFORM', {}).get('ALPHA', 120),
            sigma=aug_cfg.get('ELASTIC_TRANSFORM', {}).get('SIGMA', 120 * 0.05),
            alpha_affine=aug_cfg.get('ELASTIC_TRANSFORM', {}).get('ALPHA_AFFINE', 120 * 0.03),
            p=aug_cfg.get('ELASTIC_TRANSFORM', {}).get('P', 0.3)
        ) if aug_cfg.get('ELASTIC_TRANSFORM', {}).get('ENABLED', True) else A.NoOp(),
        
        A.GridDistortion(
            num_steps=aug_cfg.get('GRID_DISTORTION', {}).get('NUM_STEPS', 5),
            distort_limit=aug_cfg.get('GRID_DISTORTION', {}).get('DISTORT_LIMIT', 0.3),
            p=aug_cfg.get('GRID_DISTORTION', {}).get('P', 0.3)
        ) if aug_cfg.get('GRID_DISTORTION', {}).get('ENABLED', True) else A.NoOp(),
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
    ])
    
    return train_transform, val_transform


def generate_dataset(args, cfg):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    # Setup transforms
    train_transform, val_transform = get_transforms(cfg)
    
    # Setup datasets
    train_dataset = XRayDataset(
        cfg=cfg,
        is_train=True,
        transforms=train_transform
    )
    
    val_dataset = XRayDataset(
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
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg['DATASET']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['DATASET']['NUM_WORKERS'],  # Prevent memory issues during validation
        drop_last=False
    )

    # test_loader = torch.utils.data.DataLoader(
    #     test_ds, batch_size=args.batch_size, shuffle=(test_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    # )
    test_loader = None

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler


def generate_test_loader(key, args):
    key = [key]
    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        test_ds = AcdcDataset(keys=key, mode='val', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    )

    return test_loader


def generate_contrast_dataset(args):
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)

    if args.dataset == 'acdc' or args.dataset == 'ACDC':
        args.img_size = 224
        train_ds = AcdcDataset(keys=tr_keys, mode='contrast', args=args)
        val_ds = AcdcDataset(keys=val_keys, mode='contrast', args=args)
    else:
        raise NotImplementedError("dataset is not supported:", args.dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )

    return train_loader, val_loader
