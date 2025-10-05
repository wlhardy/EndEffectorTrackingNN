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

# MLP Head Module
class EndEffectorPosePredToken(nn.Module):
    def __init__(self, backbone, num_classes_joint, nbr_classes_xy, nbr_tokens=3):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        embed_dim = backbone.embed_dim  # usually 1024 or 768
        self.nbr_tokens = nbr_tokens
        
        self.learnable_tokens = nn.Parameter(torch.randn(1, self.nbr_tokens, embed_dim))

        self.class_pos_embed = nn.Parameter(
            torch.zeros(1, self.nbr_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.class_pos_embed, std=0.02)
        nn.init.normal_(self.learnable_tokens, std=1e-6)

        self.base_joint_head = nn.Linear(embed_dim, num_classes_joint // 2)
        self.base_x_head = nn.Linear(embed_dim, nbr_classes_xy)
        self.base_y_head = nn.Linear(embed_dim, nbr_classes_xy)
        
        self.norm = backbone.norm

    def interpolate_pos_encoding(self, x, w, h):
        # Taken and adapted from DINO:
        previous_dtype = x.dtype
        npatch = x.shape[1] - self.nbr_tokens
        N = self.backbone.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.backbone.pos_embed
        pos_embed = self.backbone.pos_embed.float()
        class_pos_embed = x[:, :self.nbr_tokens]
        patch_pos_embed = x[:, self.nbr_tokens:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.backbone.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.backbone.interpolate_offset) / M
            sy = float(h0 + self.backbone.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.backbone.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def forward(self, x):
        B, C, W, H = x.shape
        x = self.backbone.patch_embed(x)
        tokens = self.learnable_tokens.expand(B, -1, -1)
        x = torch.cat((tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.norm(x)
        base_joint_task_token_out = x[:, :1]
        base_joint_token_mean = base_joint_task_token_out.mean(dim=1)
        base_joint_logits = self.base_joint_head(base_joint_token_mean)
        base_x_token_out = x[:, 1:2]
        base_x_token_mean = base_x_token_out.mean(dim=1)
        base_x_logits = self.base_x_head(base_x_token_mean)
        base_y_token_out = x[:, 2:3]
        base_y_token_mean = base_y_token_out.mean(dim=1)
        base_y_logits = self.base_y_head(base_y_token_mean)
        return base_joint_logits, base_x_logits, base_y_logits