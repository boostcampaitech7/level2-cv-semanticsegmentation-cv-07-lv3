import os
import time
import wandb
import torch
import shutil
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict, Tuple
from datetime import timedelta
from utils.resources import CLASSES
from utils.utils_for_visualizer import encode_mask_to_rle
from utils.resources import CLASSES


class Trainer:
    """Trainer class for model training and validation
    
    Args:
        cfg: Configuration object containing training parameters
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config_name: Name of the configuration file
    """
    def __init__(
        self,
        cfg: Dict,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        config_name: str = 'default'
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup directories
        self.saved_dir = cfg['TRAIN']['SAVED_DIR']
        os.makedirs(self.saved_dir, exist_ok=True)
        
        self.threshold = cfg['VALIDATION']['THRESHOLD']  # Default threshold of 0.5
        
        # Setup val_log directory
        self.val_log_dir = os.path.join('val_log')
        os.makedirs(self.val_log_dir, exist_ok=True)
        
        self.config_name = config_name
        
    def train(self) -> None:
        """Main training loop"""
        self.model.train()
        print(f'Start training..')
        
        best_dice = 0.
        lowest_loss = 0.016
        best_epoch = -1
        best_model_path = None
        
        for epoch in range(self.cfg['TRAIN']['NUM_EPOCHS']):
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Training phase
            train_loss = self._train_epoch(epoch)
            
            # Step scheduler for epoch-based schedulers
            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f'Epoch {epoch+1}: Learning rate changed from {current_lr:.2e} to {new_lr:.2e}')
            
            # wandb logging
            train_log_dict = {
                "train_epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "learning_rate": current_lr
            }
            wandb.log(train_log_dict)
            
            # Validation phase
            if (epoch + 1) % self.cfg['TRAIN']['VAL_EVERY'] == 0 or train_loss < lowest_loss:

                if train_loss < lowest_loss:
                    lowest_loss = train_loss

                dice, class_dice_dict, val_loss, val_results = self._validate_epoch(epoch + 1)
                
                # Step scheduler for ReduceLROnPlateau
                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    prev_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != prev_lr:
                        print(f'Epoch {epoch+1}: Learning rate changed from {prev_lr:.2e} to {new_lr:.2e} (ReduceLROnPlateau)')
                
                # wandb validation logging
                val_log_dict = {
                    "val_epoch": epoch + 1,
                    "val_dice": dice,
                    "val_loss": val_loss
                }
                wandb.log(val_log_dict)
                
                # Only save model if dice score is above 0.93 and better than previous best
                if dice > 0.93 and best_dice < dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    
                    # Create config-specific directory
                    config_save_dir = os.path.join(self.saved_dir, self.config_name)
                    os.makedirs(config_save_dir, exist_ok=True)
                    
                    # Save model with config name and epoch number
                    model_filename = f"{self.config_name}_epoch{epoch+1}.pt"
                    model_path = os.path.join(config_save_dir, model_filename)
                    print(f"Save model in {config_save_dir} as {model_filename}")
                    
                    best_dice = dice
                    best_epoch = epoch + 1
                    best_model_path = model_path
                    self.save_model(model_path)
                    
                    # Save validation results to CSV only for best model
                    self._save_validation_results(val_results, epoch + 1)
                    
                    # Save best dice scores
                    val_log_file = os.path.join(self.val_log_dir, f'{self.config_name}.txt')
                    with open(val_log_file, 'w') as f:
                        f.write(f"Best model performance at epoch {epoch + 1}\n")
                        f.write(f"Average Dice Score: {dice:.4f}\n\n")
                        f.write("Per-class Dice Scores:\n")
                        for class_name, score in class_dice_dict.items():
                            f.write(f"{class_name}: {score:.4f}\n")
        
        # After training, copy the best model to final version if exists
        if best_model_path is not None:
            final_model_path = os.path.join(self.saved_dir, self.config_name, f"{self.config_name}.pt")
            print(f"\nSaving final best model (from epoch {best_epoch}) as {final_model_path}")
            shutil.copy2(best_model_path, final_model_path)
            
            # Update validation log to indicate final best model
            val_log_file = os.path.join(self.val_log_dir, f'{self.config_name}.txt')
            with open(val_log_file, 'a') as f:
                f.write(f"\nFinal best model saved as {self.config_name}.pt (from epoch {best_epoch})")
    
    def _train_epoch(self, epoch: int) -> float:
        """Training loop for one epoch
        
        Args:
            epoch: Current epoch number
        """
        total_loss = 0

        with tqdm(total=len(self.train_loader), desc=f'[Training Epoch {epoch+1}]', disable=False) as pbar:
            for step, data in enumerate(self.train_loader):        
                images = data[0].to(self.device)
                masks = data[1].to(self.device)
                
                # Forward pass
                outputs = self.model(images)['out']
                
                # Calculate loss
                if self.cfg['LOSS']['NAME'] == 'boundary':
                    dist_maps = data[2].to(self.device)
                    loss = self.criterion(outputs, masks, dist_maps)
                else:
                    loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    loss=f'{loss.item():.4f}',
                    avg_loss=f'{total_loss/(step+1):.4f}'
                )
        
        epoch_loss = total_loss / len(self.train_loader)
        return epoch_loss
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict, float, Dict]:
        """Validation loop for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple containing:
            - Average Dice coefficient across all classes
            - Dictionary of per-class Dice scores
            - Average validation loss
            - Validation results
        """
        val_start = time.time()
        self.model.eval()
        
        total_loss = 0
        dices = []

        rles = []
        filename_and_class = []
        
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'[Validation Epoch {epoch}]', disable=False) as pbar:
                for step, data in enumerate(self.val_loader):
                    images = data[0].to(self.device)
                    masks = data[1].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)['out']
                    
                    # Handle different output sizes
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    # Calculate loss
                    if self.cfg['LOSS']['NAME'] == 'boundary':
                        dist_maps = data[2].to(self.device)
                        loss = self.criterion(outputs['out'], masks, dist_maps)
                    else:
                        loss = self.criterion(outputs['out'], masks)
                    total_loss += loss.item()
                    
                    # Calculate Dice coefficient on GPU
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > self.threshold)

                    dice = self.dice_coef(outputs, masks)
                    dices.append(dice.detach().cpu())

                    batch_start = step * self.val_loader.batch_size

                    for i in range(outputs.size(0)):
                        current_image_name = os.path.basename(self.val_loader.dataset.filenames[batch_start + i])

                        for class_index, class_name in enumerate(CLASSES):
                            output_mask = outputs[i, class_index].detach().cpu().numpy().astype(np.uint8)
                            rle = encode_mask_to_rle(output_mask)
                            rles.append(rle)
                            filename_and_class.append(f"{current_image_name}_{class_name}")
                    
                    pbar.update(1)
                    pbar.set_postfix(dice=torch.mean(dice).item(), loss=loss.item())
        
        val_time = time.time() - val_start
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        
        # Print detailed results
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(self.cfg['CLASSES'], dices_per_class)
        ]
        print("\n".join(dice_str))
        
        avg_dice = torch.mean(dices_per_class).item()
        avg_loss = total_loss / len(self.val_loader)
        
        print(f"Average Dice: {avg_dice:.4f}")
        print(f"Validation Loss: {avg_loss:.4f} || Elapsed time: {timedelta(seconds=val_time)}\n")
        
        # Create per-class dice score dictionary
        class_dice_dict = {
            f"{c}'s dice score": d.item() 
            for c, d in zip(self.cfg['CLASSES'], dices_per_class)
        }
        
        # Split filename_and_class into separate lists
        filename, classes = zip(*[x.rsplit("_", 1) for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]
        
        # Instead of saving CSV here, return the necessary data
        val_results = {
            'image_name': image_name,
            'classes': classes,
            'rles': rles,
        }
        
        return avg_dice, class_dice_dict, avg_loss, val_results
    
    @staticmethod
    def dice_coef(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate Dice coefficient
        
        Args:
            y_pred: Predicted masks
            y_true: Ground truth masks
            
        Returns:
            Dice coefficient for each class
        """
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)
        
        eps = 0.0001
        return (2. * intersection + eps) / (
            torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps
        )
    
    def save_model(self, filepath: str) -> None:
        """Save model checkpoint
        
        Args:
            filepath: Full path to save the checkpoint file
        """
        torch.save(self.model, filepath)
    
    def _save_validation_results(self, val_results: Dict, epoch: int) -> None:
        """Save validation results to CSV
        
        Args:
            val_results: Dictionary containing validation results
            epoch: Current epoch number
        """
        # Create base validation result directory
        base_path = '../validation_result'
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        
        # Create config-specific directory
        config_path = os.path.join(base_path, self.config_name)
        if not os.path.exists(config_path):
            os.makedirs(config_path, exist_ok=True)
        
        output_csv = f'val_epoch_{epoch:02d}.csv'
        
        # Complete output path
        output_path = os.path.join(config_path, output_csv)
        
        # Save to CSV
        df = pd.DataFrame({
            "image_name": val_results['image_name'],
            "class": val_results['classes'],
            "rle": val_results['rles'],
        })
        
        df.to_csv(output_path, index=False)