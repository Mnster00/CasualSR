import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import clip
from .degradation import random_degradation

class SRDataset(Dataset):
    def __init__(self, root_dir, patch_size=192, scale=4, is_train=True):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train
        
        # Get image paths
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
            
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        hr_img = Image.open(img_path).convert('RGB')
        
        # Random crop in training
        if self.is_train:
            # Random crop
            w, h = hr_img.size
            x = random.randint(0, max(0, w - self.patch_size))
            y = random.randint(0, max(0, h - self.patch_size))
            hr_img = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
            
            # Random flip
            if random.random() < 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random rotation
            if random.random() < 0.5:
                hr_img = hr_img.rotate(90)
        
        # Get CLIP features
        with torch.no_grad():
            clip_img = self.clip_preprocess(hr_img).unsqueeze(0)
            clip_features = self.clip_model.encode_image(clip_img).squeeze(0)
        
        # Generate LR image with random degradation
        hr_tensor = self.transform(hr_img)
        lr_tensor, degradation_params = random_degradation(hr_tensor, self.scale)
        
        # Generate counterfactual sample
        cf_params = {k: v + torch.randn_like(v) * 0.1 for k, v in degradation_params.items()}
        lr_cf_tensor = random_degradation(hr_tensor, self.scale, cf_params)[0]
        
        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'lr_cf': lr_cf_tensor,
            'clip_features': clip_features,
            'degradation_params': degradation_params
        }
