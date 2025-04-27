import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from net import Restormer_Encoder, Restormer_Decoder
from utils.dataset import WatermarkDataset
import torchvision.transforms.functional as TF
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# --- Config ---
patch_h, patch_w = 224, 480
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "models/WM_04-18-00-58mb2.pth"  # путь к обученной модели
dataset = WatermarkDataset(
    root_dir=r"C:/Users/user/deep-image-prior/data/FullNS/dlnetEncoder32_9_40_alpha20",
    num_dirs=3, num_igm=500
)
transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# --- Load model ---
encoder = Restormer_Encoder().to(device)
decoder = Restormer_Decoder(dim=64).to(device)

state = torch.load(ckpt_path, map_location=device)
encoder.load_state_dict(state["encoder"])
decoder.load_state_dict(state["decoder"])
encoder.eval()
decoder.eval()
compressed_h = 0
restored_h = 0
# --- Infer ---
for comp_path, target_path, name in dataset.pairs:
    with Image.open(comp_path).convert("RGB") as img_comp, \
         Image.open(target_path).convert("RGB") as img_target:

        # Преобразуем изображения в тензоры
        img_tensor = transform(img_comp).unsqueeze(0).to(device)  # [1, 3, H, W]
        gt_tensor = transform(img_target).unsqueeze(0).to(device)

        _, _, h, w = img_tensor.shape

        # Инициализируем результат копией исходного (сжатого) изображения
        result = img_tensor.clone()

        for top in range(0, h - patch_h + 1, patch_h):
            for left in range(0, w - patch_w + 1, patch_w):
                patch = img_tensor[:, :, top:top+patch_h, left:left+patch_w]

                with torch.no_grad():
                    features = encoder(patch)
                    output = decoder(patch, features)

                result[:, :, top:top+patch_h, left:left+patch_w] = output

        # Ограничим значения
        result = result.clamp(0, 1)

        # --- Save ---
        save_image(result, f"infered_images/{name}")
        # save_image(gt_tensor, f"infered_images/gt_{name}")
        # save_image(img_tensor, f"infered_images/input_{name}")

        # --- PSNR ---

        result_np = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        gt_np = gt_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        compressed_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        psnr_compressed = compare_psnr(gt_np, compressed_np, data_range=1.0)

        # PSNR восстановленного к оригиналу (уже был)
        psnr_restored = compare_psnr(gt_np, result_np, data_range=1.0)
        restored_h += psnr_restored
        compressed_h += psnr_compressed
        print(f"{name}: PSNR restored = {psnr_restored:.2f} dB, compressed = {psnr_compressed:.2f} dB, diff = {psnr_restored - psnr_compressed}")
print(f"avg restored: {restored_h / len(dataset.pairs)}, avg compressed: {compressed_h / len(dataset.pairs)}")