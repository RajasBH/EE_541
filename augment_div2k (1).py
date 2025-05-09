"""
augment_div2k.py
Offline augmentation: saves 6 augmented PNGs for each HR/LR pair.
To run this file, download the zip files from the DIV2K website and unzip them to the location of your choice and change the BASE_DIR path.
Run:  python augment_div2k.py
"""



import cv2, random, numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = Path("DIV2K_raw")                         # adjust if needed
HR_DIR   = BASE_DIR/"DIV2K_train_HR"
LR_DIR   = BASE_DIR/"DIV2K_train_LR_bicubic"/"X2"      

OUT_HR   = BASE_DIR/"Augmented"/"HR"
OUT_LR   = BASE_DIR/"Augmented"/"LR"
for p in (OUT_HR, OUT_LR): p.mkdir(parents=True, exist_ok=True)

# -------------- Augment ops --------------
def rotate(img):
    angle=random.choice([90,180,270])
    flag={90:cv2.ROTATE_90_CLOCKWISE,
          180:cv2.ROTATE_180,
          270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]
    return cv2.rotate(img,flag)

def shear(img,f):
    h,w=img.shape[:2]
    M=np.array([[1,f,0],[0,1,0]],dtype=np.float32)
    return cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT)

def flip(img): return cv2.flip(img,1)
def crop(img,size):
    h,w=img.shape[:2]
    if h<size or w<size: return cv2.resize(img,(size,size))
    x,y=random.randint(0,h-size),random.randint(0,w-size)
    return img[x:x+size,y:y+size]

def gaussian(img,mean=50,std=50):
    noise=np.random.normal(mean,std,img.shape)
    return np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)

AUGS=[("_1",rotate),
      ("_2",lambda x:shear(x, 0.2)),
      ("_3",lambda x:shear(x,-0.2)),
      ("_4",flip),
      ("_5",lambda x:crop(x,512 if x.shape[0]>256 else 128)),
      ("_6",gaussian)]

# -------------- Main loop ---------------
hr_files=sorted(HR_DIR.glob("*.png"))
for hr_path in tqdm(hr_files,desc="Augment"):
    lr_path=LR_DIR/f"{hr_path.stem}x2.png"
    hr=cv2.imread(str(hr_path)); lr=cv2.imread(str(lr_path))
    if hr is None or lr is None: continue
    for tag,fn in AUGS:
        cv2.imwrite(str(OUT_HR/f"{hr_path.stem}{tag}.png"), fn(hr))
        cv2.imwrite(str(OUT_LR/f"{hr_path.stem}{tag}.png"), fn(lr))

print("augmentation done. Files saved to", OUT_HR, OUT_LR)
