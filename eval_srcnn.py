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
        self.lr_resize = transforms.Resize(patch_size * scale, interpolation=Image.BICUBIC)
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
        lr_img_up = self.lr_resize(transforms.ToPILImage()(lr_tensor))
        lr_tensor_up = transforms.ToTensor()(lr_img_up)
        return lr_tensor_up, hr_tensor

# === SRCNN Model ===
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)


# === Paths ===
model_path = r"C:\Users\Aren\Desktop\srcnn model\best_model.pth"
val_hr_dir = r"D:\SUPERCV\DIV2K_raw\Validation_Set\DIV2K_valid_HR"

# === Dataset & Loader ===
val_dataset = DIV2KSuperResDataset(hr_dir=val_hr_dir, scale=4, patch_size=128)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Load Model ===
model = SRCNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# === Evaluation ===
total_psnr = 0.0
total_ssim = 0.0
count = 0

with torch.no_grad():
    for lr_up, hr in tqdm(val_loader, desc="Evaluating SRCNN"):
        lr_up, hr = lr_up.to(device), hr.to(device)
        sr = torch.clamp(model(lr_up), 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        total_psnr += psnr(sr, hr, data_range=1.0).item()
        total_ssim += ssim(sr, hr, data_range=1.0).item()
        count += 1

# === Results ===
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count
print(f"\n✅ SRCNN Evaluation Complete:")
print(f"Avg PSNR: {avg_psnr:.2f} dB")
print(f"Avg SSIM: {avg_ssim:.4f}")
