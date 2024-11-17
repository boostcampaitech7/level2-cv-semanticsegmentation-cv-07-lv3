import os
import cv2
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader

from src.dataset import XRayInferenceDataset
from utils.utils_for_visualizer import encode_mask_to_rle, decode_rle_to_mask


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Inference segmentation model')
    parser.add_argument('-c', '--config', type=str, default='smp_unetplusplus_efficientb0.yaml',
                        help='path to config file')
    parser.add_argument('-m', '--model_path', type=str, default='best_model.pt',
                        help='path to model checkpoint')
    parser.add_argument('-o', '--output_path', type=str, default='output.csv',
                        help='path to save prediction results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold for binary prediction')
    args = parser.parse_args()
    
    # Add checkpoints directory to model path if not already specified
    if not os.path.dirname(args.model_path):
        args.model_path = os.path.join('checkpoints', args.model_path)
    
    # Add results directory to output path if not already specified
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist
    if not os.path.dirname(args.output_path):
        args.output_path = os.path.join(results_dir, args.output_path)
    
    return args


def load_config(config_name):
    """Load config file"""
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


class Inferencer:
    def __init__(self, cfg, model_path, threshold=0.5):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Load model
        self.model = torch.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(cfg['DATASET']['IMAGE_SIZE'], cfg['DATASET']['IMAGE_SIZE']),
        ])
        
    def predict(self, data_loader):
        """Run inference on the given data loader"""
        rles = []
        filename_and_class = []
        
        with torch.no_grad():
            for images, image_names in tqdm(data_loader, total=len(data_loader)):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                # Handle dictionary output - get the main prediction
                if isinstance(outputs, dict):
                    outputs = outputs['out']  # or the appropriate key for your model's output
                
                # Resize to original size
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > self.threshold).detach().cpu().numpy()
                
                # Convert to RLE format
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{self.cfg['CLASSES'][c]}_{image_name}")
        
        return rles, filename_and_class


def main():
    """Main inference function"""
    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup dataset and dataloader
    test_dataset = XRayInferenceDataset(
        cfg=cfg,
        transforms=A.Compose([
            A.Resize(cfg['DATASET']['IMAGE_SIZE'], cfg['DATASET']['IMAGE_SIZE']),
        ])
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg['DATASET']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['DATASET']['NUM_WORKERS'],
        drop_last=False
    )
    
    # Setup inferencer
    inferencer = Inferencer(
        cfg=cfg,
        model_path=args.model_path,
        threshold=args.threshold
    )
    
    # Run inference
    print("Starting inference...")
    rles, filename_and_class = inferencer.predict(test_loader)
    
    # Prepare submission
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    # Create submission dataframe
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    
    # Save results
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")


if __name__ == '__main__':
    main()