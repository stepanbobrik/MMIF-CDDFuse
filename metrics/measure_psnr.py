import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def evaluate_psnr_with_skimage(restored_root: str):
    """
    Проходит по папкам dir_001…dir_020 в restored_root и оригинальным
    папкам с теми же именами, вычисляет PSNR через skimage и
    возвращает средний, минимальный и максимальный PSNR.
    """
    original_root = os.path.join(
        '..', 'FullNS', 'dlnetEncoder32_9_40_alpha20',
        'watermarked'
    )
    # original_root = os.path.join(
    #     '..', 'FullNS', 'cover'
    # )

    psnr_values = []

    for idx in range(1, 21):
        dir_name = f'dir_{idx:03d}'
        restored_dir = os.path.join(restored_root, dir_name)
        original_dir = os.path.join(original_root, dir_name)

        if not os.path.isdir(restored_dir) or not os.path.isdir(original_dir):
            print(f"[SKIP] Пропущена директория: {dir_name}")
            continue

        for fname in os.listdir(restored_dir):
            if not fname.lower().endswith('.png'):
                continue

            restored_path = os.path.join(restored_dir, fname)
            original_path = os.path.join(original_dir, fname)

            if not os.path.isfile(original_path):
                print(f"[WARN] Оригинал не найден для {fname}")
                continue

            # Загрузка изображений и конвертация в NumPy-массивы
            with Image.open(restored_path) as im1, Image.open(original_path) as im2:
                arr1 = np.array(im1.convert('RGB'))
                arr2 = np.array(im2.convert('RGB'))

            # Усечение до минимального общего размера, если нужно
            if arr1.shape != arr2.shape:
                h, w = min(arr1.shape[0], arr2.shape[0]), min(arr1.shape[1], arr2.shape[1])
                arr1 = arr1[:h, :w]
                arr2 = arr2[:h, :w]
                print(f"[SIZE] Усечено до {h}x{w} для {fname}")

            # Вычисляем PSNR через skimage
            psnr_val = compare_psnr(arr2, arr1, data_range=255)
            print(f"{fname}: PSNR = {psnr_val:.2f} dB")
            psnr_values.append(psnr_val)

    if not psnr_values:
        print("Нет ни одной корректной пары изображений для оценки.")
        return None

    avg = np.mean(psnr_values)
    mx  = np.max(psnr_values)
    mn  = np.min(psnr_values)

    print(f"\nРезультаты по всем файлам: avg: {avg:.2f} dB max: {mx:.2f} dB min: {mn:.2f} dB")

    return avg

if __name__ == "__main__":
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20','watermarked_lab', 'jpeg50', 'attacked_images')) #avg 36.52 max 48.18 min 29.52
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab', 'jpeg50', 'attacked_images')) #dir_001 avg 36.63 max 43.59 min 29.52 | cover 37.50 max 43.52 min 30.56
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab', 'jpeg50', 'attacked_images')) #dir_002 avg: 39.56 dB max: 45.21 dB min: 32.23 dB
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab', 'jpeg50', 'attacked_images')) #dir_003 avg: 40.10 dB max: 44.61 dB min: 34.69 dB
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'restored')) # avg 36.01 max 44.52 min 28.74 | cover avg 37.02 max 44.77 min 29.61
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'resotred_mmif_cddfuse')) #dir_001 avg: 36.62 dB max: 43.77 dB min: 29.57 dB
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'resotred_mmif_cddfuse')) #dir_002 avg: 39.53 dB max: 45.09 dB min: 32.34 dB
   # evaluate_psnr_with_skimage(os.path.join('..', 'FullNS', 'resotred_mmif_cddfuse')) #dir_003 avg: 40.05 dB max: 44.48 dB min: 34.69 dB | avg: 39.92 dB max: 44.48 dB min: 34.69 dB
   pass