import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn.init as init
import math

TOKEN_LIST = ["single_token"]

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


class EndEffectorPosePredToken(nn.Module):
    """
    Single-token variant: Uses 1 learnable token instead of 3,
    but maintains 3 separate prediction heads for base_joint, base_x, and base_y.
    """
    def __init__(self, backbone, nbr_tokens=1):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        embed_dim = backbone.embed_dim  # usually 1024 or 768
        self.nbr_tokens = nbr_tokens  # Always 1 for this variant
        
        self.learnable_tokens = nn.Parameter(torch.randn(1, self.nbr_tokens, embed_dim))

        nn.init.normal_(self.learnable_tokens, std=0.02)

        # Three separate heads, all operating on the same single token
        self.base_joint_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2),
        )
        self.base_x_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        self.base_y_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.norm = self.backbone.norm


    def prepare_tokens(self, x):
        x = self.backbone.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        learnable_tokens = self.learnable_tokens + 0 * self.backbone.mask_token

        if self.backbone.n_storage_tokens > 0:
            storage_tokens = self.backbone.storage_tokens
        else:
            storage_tokens = torch.empty(
                    1,
                    0,
                    learnable_tokens.shape[-1],
                    dtype=learnable_tokens.dtype,
                    device=learnable_tokens.device,
            )

        x = torch.cat(
            [
                learnable_tokens.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        
        return x, (H, W)
    
    
    def forward_feature(self, x):
        x, rope = self.prepare_tokens(x)
        
        for _, blk in enumerate(self.backbone.blocks):
            if self.backbone.rope_embed is not None:
                H, W = rope
                rope_sincos = self.backbone.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, x in enumerate(all_x):
            x_norm = self.norm(x)
            x_norm_cls_reg = x_norm[:self.backbone.n_storage_tokens + self.nbr_tokens, :]
            x_norm_patch = x_norm[self.backbone.n_storage_tokens + self.nbr_tokens:, :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:self.nbr_tokens, :],
                    "x_storage_tokens": x_norm_cls_reg[self.nbr_tokens:, :],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x
                }
            )
        return output
    

    def forward(self, x):
        output = self.forward_feature(x)
        x_norm_clstoken = [d["x_norm_clstoken"] for d in output]
        x_norm_clstoken = torch.stack(x_norm_clstoken, dim=0)

        # All three heads operate on the same single token
        single_token = x_norm_clstoken[:, 0:1, :]  # Shape: (batch, 1, embed_dim)
        
        base_joint_sincos = self.base_joint_head(single_token).squeeze(1)
        base_x = self.base_x_head(single_token).squeeze()
        base_y = self.base_y_head(single_token).squeeze()
        
        return base_joint_sincos, base_x, base_y
