import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from piq import psnr, ssim
from tqdm import tqdm
import torch.nn as nn

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset Class ===
class DIV2KSuperResDataset(Dataset):
    def __init__(self, hr_dir, scale=4, patch_size=128):
        self.hr_files = sorted(Path(hr_dir).glob("*.png"))
        self.scale = scale
        self.patch_size = patch_size
        self.hr_transform = transforms.Compose([
            transforms.CenterCrop(patch_size * scale),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(patch_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_files[idx]).convert("RGB")
        hr_tensor = self.hr_transform(hr_img)
        lr_tensor = self.lr_transform(hr_tensor)
        return lr_tensor, hr_tensor

# === 12-layer DnCNN Model ===
class DnCNN(nn.Module):
    def __init__(self, channels=3):
        super(DnCNN, self).__init__()
        layers = []

        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(10):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)

# === Paths ===
model_path = r"C:\Users\Aren\Desktop\dcnn model\best_model.pth"
val_hr_dir = r"D:\SUPERCV\DIV2K_raw\Validation_Set\DIV2K_valid_HR"

# === Dataset & Dataloader ===
val_dataset = DIV2KSuperResDataset(hr_dir=val_hr_dir, scale=4, patch_size=128)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Load model ===
model = DnCNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# === Evaluation Loop ===
total_psnr = 0.0
total_ssim = 0.0
count = 0

with torch.no_grad():
    for lr, hr in tqdm(val_loader, desc="Evaluating DnCNN"):
        lr, hr = lr.to(device), hr.to(device)

        # Predict & clamp SR output
        sr = model(lr)
        sr = torch.clamp(sr, 0.0, 1.0)

        # Resize & clamp HR ground truth
        hr = torch.nn.functional.interpolate(hr, size=sr.shape[-2:], mode='bicubic', align_corners=False)
        hr = torch.clamp(hr, 0.0, 1.0)

        # Evaluate
        total_psnr += psnr(sr, hr, data_range=1.0).item()
        total_ssim += ssim(sr, hr, data_range=1.0).item()
        count += 1

# === Results ===
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print(f"\n✅ DnCNN Evaluation Complete:")
print(f"Avg PSNR: {avg_psnr:.2f} dB")
print(f"Avg SSIM: {avg_ssim:.4f}")
