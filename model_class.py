import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn.init as init
import math

TOKEN_LIST = ["joint3", "base_x", "base_y"]
#[
    #"joint1", "joint2", "joint3", "joint4",
    #"base_x", "base_y", 
    #"left_claw_x", "left_claw_y", 
    #"right_claw_x", "right_claw_y"
#]

# Utility
def center_pad_to_multiple(x, multiple):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x_padded = TF.pad(x, [pad_left, pad_top, pad_right, pad_bottom])
    return x_padded

def load_image_as_tensor(image_path, rotate=False):
    image = Image.open(image_path).convert("RGB")
    if rotate:
        image = image.rotate(180)
    transform = T.Compose([
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # [B, C, H, W]


class EndEffectorPosePredClass(nn.Module):
    def __init__(self, backbone, num_classes_joint, nbr_classes_xy, nbr_tokens=1):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        embed_dim = backbone.embed_dim  # usually 1024 or 768
        hidden_dim = 1024
        self.nbr_tokens = nbr_tokens
        self.fixed_token_size = 2974
        self.patch_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.base_joint_head = nn.Linear(embed_dim, num_classes_joint // 2)
        self.base_x_head = nn.Linear(embed_dim, nbr_classes_xy)
        self.base_y_head = nn.Linear(embed_dim, nbr_classes_xy)

    def forward(self, x):
        B, C, W, H = x.shape
        x = self.backbone.forward_features(x)
        x = torch.cat([x['x_norm_regtokens'], x['x_norm_patchtokens']], dim=1)
        x = self.patch_mlp(x)
        x = x.mean(dim=1)

        base_joint_logits = self.base_joint_head(x).squeeze(1)
        base_x_logits = self.base_x_head(x).squeeze(1)
        base_y_logits = self.base_y_head(x).squeeze(1)

        return base_joint_logits, base_x_logits, base_y_logits