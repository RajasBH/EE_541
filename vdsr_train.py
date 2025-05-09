import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim
import cv2, random, numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
BASE = Path("DIV2K_raw")
SCALE = 4
EPOCHS = 10
BATCH_SIZE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERCEPTUAL_WEIGHT = 0.05

# -------------------------------
# Dataset Class
# -------------------------------
class VDSRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=128, noise_std=15, augment=True,
                 hr_size=(1024, 1024), lr_size=(256, 256)):
        self.hr_paths = sorted(hr_dir.glob("*.png"))
        self.lr_paths = sorted(lr_dir.glob("*.png"))
        self.crop = crop_size
        self.lr_crop = crop_size // SCALE
        self.noise_std = noise_std
        self.augment = augment
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = cv2.cvtColor(cv2.imread(str(self.hr_paths[idx])), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(str(self.lr_paths[idx])), cv2.COLOR_BGR2RGB)

        hr = cv2.resize(hr, self.hr_size[::-1], interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(lr, self.lr_size[::-1], interpolation=cv2.INTER_CUBIC)

        h, w = hr.shape[:2]
        x, y = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        hr_crop = hr[x:x+self.crop, y:y+self.crop]
        lr_crop = lr[x//SCALE:(x//SCALE)+self.lr_crop, y//SCALE:(y//SCALE)+self.lr_crop]

        lr_crop = np.clip(lr_crop.astype(np.float32) + np.random.normal(0, self.noise_std, lr_crop.shape), 0, 255).astype(np.uint8)

        if self.augment and random.random() < 0.5:
            hr_crop = cv2.flip(hr_crop, 1)
            lr_crop = cv2.flip(lr_crop, 1)

        return self.to_tensor(lr_crop), self.to_tensor(hr_crop)

# -------------------------------
# Model Definition
# -------------------------------
class VDSR(nn.Module):
    def __init__(self, depth=20, channels=64):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=SCALE, mode="bicubic", align_corners=False)
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 3, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        upscaled = self.upsample(x)
        out = self.body(upscaled)
        return torch.clamp(out + upscaled, 0, 1)

# -------------------------------
# Metrics
# -------------------------------
def compute_psnr(pred, target):
    return 10 * torch.log10(1 / F.mse_loss(pred, target))

def compute_ssim(pred, target):
    scores = [
        ssim(pred[i].permute(1, 2, 0).cpu().numpy(),
             target[i].permute(1, 2, 0).cpu().numpy(),
             data_range=1, channel_axis=-1)
        for i in range(pred.size(0))
    ]
    return np.mean(scores)

# -------------------------------
# VGG Perceptual Loss (Features up to relu3_3)
# -------------------------------
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:17].to(DEVICE).eval()
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(a, b):
    return F.l1_loss(vgg(a), vgg(b))

# -------------------------------
# Setup
# -------------------------------
HR_DIR = BASE / "Augmented/HR"
LR_DIR = BASE / "Augmented/LR"

train_dataset = VDSRDataset(HR_DIR, LR_DIR)
val_dataset = VDSRDataset(HR_DIR, LR_DIR, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

model = VDSR().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# -------------------------------
# Training Loop
# -------------------------------
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch}"):
        lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
        preds = model(lr_imgs)

        mse = F.mse_loss(preds, hr_imgs)
        percept = perceptual_loss(preds, hr_imgs)
        loss = mse + PERCEPTUAL_WEIGHT * percept

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    total_psnr = total_ssim = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            preds = model(lr_imgs.to(DEVICE)).cpu()
            total_psnr += compute_psnr(preds, hr_imgs)
            total_ssim += compute_ssim(preds, hr_imgs)

    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    history.append((avg_psnr, avg_ssim))

    print(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

# -------------------------------
# Save & Plot
# -------------------------------
torch.save(model.state_dict(), "vdsr_mse_perceptual.pth")

psnrs, ssims = zip(*history)
plt.plot(psnrs, label="PSNR")
plt.plot(ssims, label="SSIM")
plt.title("Validation Metrics Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
