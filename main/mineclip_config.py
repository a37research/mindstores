"""
MineCLIP Configuration

This configuration matches the MineCLIP attention variant settings.
The resolution is set to match the pretrained model's expectations.
"""

import os
import torch
from omegaconf import OmegaConf

# Base configuration for model architecture
MINECLIP_CONFIG = {
    'arch': 'vit_base_p16_fz.v2.t2',
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': 'attn.d2.nh8.glusw',  # Attention pooling
    'resolution': [160, 256]  # Match pretrained model's resolution
}

# Checkpoint configuration
CHECKPOINT_CONFIG = {
    'path': os.path.join(os.path.dirname(__file__), 'attn.pth'),
    'checksum': 'b5ece9198337cfd117a3bfbd921e56da'
}

def get_device():
    """Get the device to use for MineCLIP."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_image(img, target_resolution=(160, 256)):
    """Resize image tensor for MineCLIP."""
    return torch.nn.functional.interpolate(
        img, 
        size=target_resolution, 
        mode='bilinear', 
        align_corners=False
    )

def preprocess_obs(obs, device):
    """Preprocess observation for MineCLIP.
    Resizes the image to match the pretrained model's expected resolution.
    """
    rgb = torch.from_numpy(obs['rgb']).unsqueeze(0)
    rgb = rgb.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    rgb = resize_image(rgb, target_resolution=MINECLIP_CONFIG['resolution'])
    return rgb
