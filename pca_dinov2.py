import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import os

IMAGE_PATH = "/home/wilah/workspace/SVO_processing/output/20250410-grapple_fully_open_motion./rgb/left/1744299454884800000.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load DINOv2 backbone
backbone_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)

# Freeze backbone
for param in backbone_model.parameters():
    param.requires_grad = False

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

class DINOv2PCAProjector(nn.Module):
    def __init__(self, backbone, pca_components=3):
        super().__init__()
        self.backbone = backbone
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components)
        self.pca_fitted = False
        self.patch_size = backbone.patch_size  # Usually 14

    def forward(self, x):
        # Pad input so height and width are divisible by patch size
        x = center_pad_to_multiple(x, self.patch_size)
        x = x.to(device)
        _, _, H, W = x.shape

        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(x, n=1)[0]  # shape: (B, N, D)

        # Extract patch tokens (B, N, D)
        tokens = features
        B, N, D = tokens.shape

        # Flatten for PCA
        tokens_flat = tokens.reshape(-1, D).cpu().numpy()

        if not self.pca_fitted:
            self.pca.fit(tokens_flat)
            self.pca_fitted = True

        tokens_pca = self.pca.transform(tokens_flat)
        tokens_pca = torch.tensor(tokens_pca, dtype=torch.float32).reshape(B, N, self.pca_components).to(device)

        # Reshape to (B, C, H, W)
        output_H = int(H/self.patch_size)
        output_W = int(W/self.patch_size)
        tokens_pca = tokens_pca.permute(0, 2, 1).reshape(B, self.pca_components, output_H, output_W)
        return tokens_pca

def visualize_pca_image(pca_tensor):
    """Convert PCA tensor to a [0, 1] RGB image and display it."""
    # Normalize to [0, 1] for visualization
    img = pca_tensor[0].detach().cpu().numpy()
    img -= img.min()
    img /= img.max()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("test.png")

def load_image_as_tensor(image_path, rotate=False):
    image = Image.open(image_path).convert("RGB")
    if(rotate):
        image = image.rotate(180)
    transform = T.Compose([
        T.ToTensor(),  # Converts to [0,1] and CHW
    ])
    return transform(image).unsqueeze(0)  # Add batch dim

# Example usage with a real image
image_path = IMAGE_PATH
if os.path.exists(image_path):
    input_tensor = load_image_as_tensor(image_path, True).to(device)
    model = DINOv2PCAProjector(backbone_model, pca_components=3).to(device)
    pca_output = model(input_tensor)
    print("PCA output shape:", pca_output.shape)
    visualize_pca_image(pca_output)
else:
    print(f"Image not found at {image_path}. Please check the path.")
