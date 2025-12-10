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


class EndEffectorPosePredToken(nn.Module):
    def __init__(self, backbone, num_classes_joint, nbr_classes_xy, nbr_tokens=3):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        embed_dim = backbone.embed_dim  # usually 1024 or 768
        self.nbr_tokens = nbr_tokens
        
        self.learnable_tokens = nn.Parameter(torch.randn(1, self.nbr_tokens, embed_dim))

        nn.init.normal_(self.learnable_tokens, std=0.02)

        self.base_joint_head = nn.Linear(embed_dim, num_classes_joint // 2)
        self.base_x_head = nn.Linear(embed_dim, nbr_classes_xy)
        self.base_y_head = nn.Linear(embed_dim, nbr_classes_xy)
        
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

        base_joint_task_token_out = x_norm_clstoken[:, :1, :]
        base_joint_logits = self.base_joint_head(base_joint_task_token_out).squeeze(1)
        base_x_token_out = x_norm_clstoken[:, 1:2, :]
        base_x_logits = self.base_x_head(base_x_token_out).squeeze(1)
        base_y_token_out = x_norm_clstoken[:, 2:3, :]
        base_y_logits = self.base_y_head(base_y_token_out).squeeze(1)
        return base_joint_logits, base_x_logits, base_y_logits