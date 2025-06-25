import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn.init as init

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
class EndEffectorPosePred(nn.Module):
    def __init__(self, backbone, num_classes, mpl_dim):
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        embed_dim = backbone.embed_dim  # usually 1024 or 768
        nbr_patches_h = 77
        nbr_patches_w = 76
        intermediate_dim = 8
        self.reduce_dim = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim/2)),
            nn.ReLU(),
            nn.Linear(int(embed_dim/2), intermediate_dim)
        )

        self.joint1 = nn.Sequential(
            nn.Linear(intermediate_dim * nbr_patches_h * nbr_patches_w, mpl_dim),
            nn.ReLU(),
            nn.Linear(mpl_dim, num_classes)
        )

        self.joint2 = nn.Sequential(
            nn.Linear(intermediate_dim * nbr_patches_h * nbr_patches_w, mpl_dim),
            nn.ReLU(),
            nn.Linear(mpl_dim, num_classes)
        )
        
        self.joint3 = nn.Sequential(
            nn.Linear(intermediate_dim * nbr_patches_h * nbr_patches_w, mpl_dim),
            nn.ReLU(),
            nn.Linear(mpl_dim, int(num_classes / 2))
        )
        
        self.joint4 = nn.Sequential(
            nn.Linear(intermediate_dim * nbr_patches_h * nbr_patches_w, mpl_dim),
            nn.ReLU(),
            nn.Linear(mpl_dim, num_classes)
        )

        self.base_x = nn.Sequential(
            nn.Linear(intermediate_dim * nbr_patches_h * nbr_patches_w, mpl_dim),
            nn.ReLU(),
            nn.Linear(mpl_dim, 1),
            nn.Sigmoid()
        )

        self.base_y = nn.Sequential(
            nn.Linear(intermediate_dim * nbr_patches_h * nbr_patches_w, mpl_dim),
            nn.ReLU(),
            nn.Linear(mpl_dim, 1),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for layer in [self.reduce_dim, self.joint1, self.joint2, self.joint3, self.joint4]:
            for m in layer:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=1)[0]

        features = self.reduce_dim(features)
        features = features.flatten(start_dim=1)

        # TODO WLH: Only predict joint 3
        j1 = self.joint1(features)
        #j2_input = torch.cat([features, j1], dim=1)
        j2 = self.joint2(features)
        #j3_input = torch.cat([features, j1, j2], dim=1)
        j3 = self.joint3(features)
        #j4_input = torch.cat([features, j1, j2, j3], dim=1)
        j4 = self.joint4(features)
        base_x = self.base_x(features)
        base_y = self.base_y(features)
        return j1, j2, j3, j4, base_x, base_y