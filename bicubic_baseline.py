"""

No training – upsamples LR with bicubic and reports PSNR / SSIM.
Plots 3 random examples.
"""

import torch, cv2, random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------- Paths ----------
BASE = Path("DIV2K_raw")
HR_DIR = BASE/"Augmented/HR"            # use augmented set
LR_DIR = BASE/"Augmented/LR"

SCALE  = 4

# ---------- Dataset ----------
class PairDS(Dataset):
    def __init__(self, hr_dir, lr_dir,hr_size=(1024,1024),lr_size=(256,256)):
        self.hr=sorted(hr_dir.glob("*.png"))
        self.lr=sorted(lr_dir.glob("*.png"))
        self.tot=ToTensor()
        self.hr_size=hr_size
        self.lr_size=lr_size
    def __len__(self): return len(self.hr)
    def __getitem__(self,i):
        hr = cv2.cvtColor(cv2.imread(str(self.hr[i])), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(str(self.lr[i])), cv2.COLOR_BGR2RGB)
        hr = cv2.resize(hr, self.hr_size[::-1], interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(lr, self.lr_size[::-1], interpolation=cv2.INTER_CUBIC)


        return self.tot(lr), self.tot(hr)

ds = PairDS(HR_DIR, LR_DIR)
loader = DataLoader(ds, batch_size=4, shuffle=False)

# ---------- Metrics ----------
def psnr(a,b): return 10*torch.log10(1/F.mse_loss(a,b))
def ssim_b(a,b):
    s=[ssim(b[i].permute(1,2,0).numpy(), a[i].permute(1,2,0).numpy(),
        channel_axis=-1,data_range=1) for i in range(a.size(0))]
    return sum(s)/len(s)

# ---------- Evaluation ----------
p=s=0;n=0
with torch.no_grad():
    for lr,hr in loader:
        up=F.interpolate(lr,scale_factor=SCALE,mode="bicubic",align_corners=False)
        p+=psnr(up,hr); s+=ssim_b(up,hr); n+=1
print(f"Bicubic baseline  PSNR={p/n:.2f} dB  SSIM={s/n:.4f}")

# ---------- Visualization ----------
def show(lr,up,hr):
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for a,t in zip([lr,up,hr],["LR","Bicubic","HR"]):
        axs[[lr,up,hr].index(a)].imshow(a.permute(1,2,0)); axs[[lr,up,hr].index(a)].set_title(t); axs[[lr,up,hr].index(a)].axis("off")
    plt.show()

for _ in range(3):
    i=random.randint(0,len(ds)-1)
    lr,hr=ds[i]; up=F.interpolate(lr.unsqueeze(0),scale_factor=SCALE,mode="bicubic",align_corners=False).squeeze()
    show(lr,up,hr)
