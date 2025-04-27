# -*- coding: utf-8 -*-
import os, sys, time, datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import kornia

from net import Restormer_Encoder, Restormer_Decoder
from utils.dataset import WatermarkDataset
import matplotlib.pyplot as plt

# === для логов по эпохам ===
epoch_losses = []
epoch_psnrs   = []

# CONFIG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Hyperparams
num_epochs          = 120
batch_size          = 1
lr                  = 1e-3
coeff_mse_loss      = 1.0
coeff_ssim_loss     = 0.0    # вес SSIM в итоговом loss
coeff_tv            = 0.1    # вес TV
# (они же могут быть отрегулированы вами позже)

# Dataset
dataset      = WatermarkDataset(
    root_dir=r"C:/Users/user/deep-image-prior/data/FullNS/dlnetEncoder32_9_40_alpha20",
    num_dirs=20, num_igm=10
)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
encoder = Restormer_Encoder().to(device)
decoder = Restormer_Decoder(dim=64).to(device)

# Optimizer & Scheduler
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=lr
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # следим за loss (минимизируем)
    factor=0.5,
    patience=5,
    verbose=True
)

# Loss functions
MSELoss   = nn.MSELoss()
SSIMLoss  = kornia.losses.ssim_loss
L1Loss    = nn.L1Loss()

# Prepare output dirs
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
Path("debug_images").mkdir(exist_ok=True)
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()

    epoch_loss = 0.0
    epoch_psnr = 0.0
    n_batches  = len(train_loader)

    for i, (x_input, x_target, name) in enumerate(train_loader, 1):
        x_input, x_target = x_input.to(device), x_target.to(device)
        optimizer.zero_grad()

        # forward
        feats  = encoder(x_input)
        output = decoder(x_input, feats)

        # compute losses
        mse  = MSELoss(output, x_target)
        ssim = SSIMLoss(x_target, output, window_size=11, reduction='mean')
        tv   = L1Loss(
                   kornia.filters.SpatialGradient()(x_input),
                   kornia.filters.SpatialGradient()(output)
               )

        loss = (
            coeff_mse_loss * mse
          + coeff_ssim_loss * ssim
          + coeff_tv       * tv
        )
        loss.backward()
        # nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_grad_norm_value)
        # nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_grad_norm_value)
        optimizer.step()

        # accumulate for reporting
        epoch_loss += loss.item()
        with torch.no_grad():
            psnr_val = 10 * torch.log10(1.0 / (mse + 1e-8))
            epoch_psnr += psnr_val.item()

        # progress bar
        sys.stdout.write(
            f"\r[Epoch {epoch+1}/{num_epochs}] "
            f"[Batch {i}/{n_batches}] "
            f"loss: {loss.item():.4f}  "
            f"PSNR: {psnr_val.item():.2f} dB"
        )

    # average per epoch
    avg_loss = epoch_loss / n_batches
    avg_psnr = epoch_psnr / n_batches

    epoch_losses.append(avg_loss)
    epoch_psnrs  .append(avg_psnr)

    # step scheduler on avg_loss
    scheduler.step(avg_loss)

    print(
        f"\n=> Epoch {epoch+1} summary: "
        f"avg_loss={avg_loss:.4f}, avg_psnr={avg_psnr:.2f} dB"
    )

    # Save debug images + plot
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        x_in, x_tg, name = next(iter(train_loader))
        x_in, x_tg = x_in.to(device), x_tg.to(device)
        out = decoder(x_in, encoder(x_in))

        # сохраняем картинки
        vutils.save_image(out, f"debug_images/epoch{epoch+1}_restored_norm_{name[0]}.png", normalize=True)
        vutils.save_image(out.clamp(0,1), f"debug_images/epoch{epoch+1}_restored_{name[0]}.png", normalize=False)
        vutils.save_image(x_tg, f"debug_images/epoch{epoch+1}_gt_{name[0]}.png", normalize=True)
        vutils.save_image(x_in, f"debug_images/epoch{epoch+1}_input_{name[0]}.png", normalize=True)

        # рисуем график
        plt.figure(figsize=(10,5))
        plt.plot(epoch_losses, label='Loss')
        plt.plot(epoch_psnrs,   label='PSNR (dB)')
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Loss & PSNR per Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("debug_images/loss_psnr_plot.png")
        plt.close()

# Сохраняем итоговую модель
Path("models").mkdir(exist_ok=True)
torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
}, f"models/WM_{timestamp}.pth")
