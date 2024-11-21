import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

class XRayDataset(Dataset):
    """X-Ray image segmentation dataset class
    
    Args:
        cfg: Configuration object containing dataset parameters
        is_train: Whether this is training dataset or not
        transforms: Albumentations transforms to apply
    """
    def __init__(
        self, 
        cfg: Dict,
        is_train: bool = True,
        transforms: Optional[object] = None
    ):
        self.cfg = cfg
        self.is_train = is_train
        self.transforms = transforms
        
        # Initialize class mappings
        self.classes = cfg['CLASSES']
        self.class2ind = {v: i for i, v in enumerate(self.classes)}
        self.ind2class = {v: k for k, v in self.class2ind.items()}
        
        # Set root paths
        self.image_root = cfg['DATASET']['IMAGE_ROOT']
        self.label_root = cfg['DATASET']['LABEL_ROOT']
        
        # Get validation fold from config
        self.K_fold = cfg['DATASET'].get('FOLD', 0)  # Default to 0 if not specified
        
        # Initialize dataset
        self.filenames, self.labelnames = self._init_dataset()

    def _init_dataset(self) -> Tuple[List[str], List[str]]:
        """Initialize dataset by finding all image and label files
        
        Returns:
            Tuple of lists containing image and label filenames
        """
        # Find all PNG files
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        # Find all JSON files
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=self.label_root)
            for root, _dirs, files in os.walk(self.label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        # Verify matching files exist
        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
        
        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
        
        # Sort files
        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        # Split train-valid
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # Group by folder to keep same person's hands together
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # Dummy labels for GroupKFold
        ys = [0 for _ in _filenames]
        
        # Use GroupKFold to split data
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                if i == self.K_fold:  # Use fold 0 as validation
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            else:
                if i == self.K_fold:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
                    break
                
        return filenames, labelnames

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset
        
        Args:
            item: Index of the item to get
            
        Returns:
            Tuple of (image, label) tensors
        """
        # Load image
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        # image = image / 255.
        image = (image - image.min()) / (image.max() - image.min())
        
        # Load label
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # Create empty label tensor
        label_shape = tuple(image.shape[:2]) + (len(self.classes),)
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # Load and process annotations
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # Fill label tensor
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])
            
            # Convert polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        # Apply transforms if specified
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # Convert to channel-first format
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label

class XRayInferenceDataset(Dataset):
    """Dataset class for inference
    
    Args:
        cfg: Configuration object containing dataset parameters
        transforms: Albumentations transforms to apply
    """
    def __init__(
        self,
        cfg: Dict,
        transforms: Optional[object] = None
    ):
        self.cfg = cfg
        self.transforms = transforms
        self.image_root = cfg['DATASET']['TEST_IMAGE_ROOT']
        
        # Find all test images
        self.filenames = self._init_dataset()
        
    def _init_dataset(self) -> List[str]:
        """Initialize dataset by finding all test images
        
        Returns:
            List of image filenames
        """
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=self.image_root)
            for root, _dirs, files in os.walk(self.image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        return sorted(pngs)
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        """Get a single item from the dataset
        
        Args:
            item: Index of the item to get
            
        Returns:
            Tuple of (image tensor, image filename)
        """
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = (image - image.min()) / (image.max() - image.min())
        # image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # Convert to channel-first format
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
            
        return image, image_name

