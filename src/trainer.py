import os
import datetime
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from repo.utils.utils_for_visualizer import encode_mask_to_rle
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
        model_name: Name of the model file to save
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
        model_name: str = 'best_model.pt'
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
        
        self.model_name = model_name
        
    def train(self) -> None:
        """Main training loop"""
        self.model.train()
        print(f'Start training..')
        
        best_dice = 0.
        
        for epoch in range(self.cfg['TRAIN']['NUM_EPOCHS']):
            # Training phase
            train_loss = self._train_epoch(epoch)

            # wandb α
            train_log_dict = {
                "train_epoch": epoch + 1,
                "train_loss": round(train_loss, 4)
            }
            wandb.log(train_log_dict)
            
            # Validation phase
            if (epoch + 1) % self.cfg['TRAIN']['VAL_EVERY'] == 0:
                dice, class_dice_dict, val_loss = self._validate_epoch(epoch + 1)
                
                # Scheduler step - validation 후에 호출
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # validation loss 사용
                else:
                    self.scheduler.step()          # 일반적인 step
                
                # wandb validation α
                val_log_dict = {
                    "val_epoch": epoch + 1,
                    "val_dice": dice,
                    "val_loss": val_loss
                }
                wandb.log(val_log_dict)
                
                if best_dice < dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    print(f"Save model in {self.saved_dir} as {self.model_name}")
                    best_dice = dice
                    self.save_model(self.model_name)
    
    def _train_epoch(self, epoch: int) -> float:
        """Training loop for one epoch
        
        Args:
            epoch: Current epoch number
        """
        total_loss = 0
        with tqdm(total=len(self.train_loader), desc=f'[Training Epoch {epoch+1}]', disable=False) as pbar:
            for step, (images, masks) in enumerate(self.train_loader):            
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs['out'], masks)
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
        
        # ReduceLROnPlateau가 아닌 스케줄러의 경우 여기서 step
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return epoch_loss
    
    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict, float]:
        """Validation loop for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple containing:
            - Average Dice coefficient across all classes
            - Dictionary of per-class Dice scores
            - Average validation loss
        """
        val_start = time.time()
        self.model.eval()
        
        total_loss = 0
        dices = []

        rles = []
        filename_and_class = []
        
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'[Validation Epoch {epoch}]', disable=False) as pbar:
                for step, (images, masks) in enumerate(self.val_loader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)['out']
                    
                    # Handle different output sizes
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)
                    
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                    # Calculate loss
                    loss = self.criterion(outputs, masks)
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
        output_path = '../validation_result'
        output_csv = f'/val_epoch_{epoch:2d}.csv'

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        filename, classes = zip(*[x.rsplit("_", 1) for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]

 
        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        })
        
        df.to_csv(output_path + output_csv, index=False)
        
        return avg_dice, class_dice_dict, avg_loss
    
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
    
    def save_model(self, filename: str) -> None:
        """Save model checkpoint
        
        Args:
            filename: Name of the checkpoint file
        """
        output_path = os.path.join(self.saved_dir, filename)
        torch.save(self.model, output_path)