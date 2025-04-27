import os
import numpy as np
import pandas as pd

def evaluate_cwv_accuracy(extracted_root_path):
    # Путь к оригинальным ЦВЗ (замени на свой при необходимости)
    original_root_path = os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarks')

    total_ber = 0
    total_files = 0
    max_ber = -1
    min_ber = 2

    for dir_index in range(1, 21):
        dir_name = f'dir_{dir_index:03d}'
        extracted_dir = os.path.join(extracted_root_path, dir_name)
        original_dir = os.path.join(original_root_path, dir_name)

        if not os.path.isdir(extracted_dir) or not os.path.isdir(original_dir):
            print(f"Пропущена директория: {dir_name}")
            continue

        for filename in os.listdir(extracted_dir):
            if not filename.endswith('.xls'):
                continue

            extracted_path = os.path.join(extracted_dir, filename)
            original_path = os.path.join(original_dir, filename)

            if not os.path.isfile(original_path):
                print(f"[!] Оригинал не найден: {original_path}")
                continue

            # Считываем и округляем извлечённые значения
            extracted_data = pd.read_excel(extracted_path, header=None).squeeze()
            extracted_bits = np.round(extracted_data).astype(int)

            # Считываем оригинальные значения
            original_bits = pd.read_excel(original_path, header=None).squeeze().astype(int)

            # Сравнение по минимальной длине
            min_len = min(len(extracted_bits), len(original_bits))
            if len(extracted_bits) != len(original_bits):
                print(f"[~] Несовпадение длины в {filename}, сравниваются первые {min_len} бит.")

            errors = np.sum(extracted_bits[:min_len] != original_bits[:min_len])
            ber = errors / min_len

            total_ber += ber
            max_ber = max(max_ber, ber)
            min_ber = min(min_ber, ber)
            print(filename, ber)
            total_files += 1

    if total_files == 0:
        print("Нет валидных пар файлов.")
        return None

    average_ber = total_ber / total_files
    print(f'avg: {average_ber:.4f} Max: {max_ber:.4f} Min: {min_ber:.4f}')
    return average_ber


if __name__ == "__main__":
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20','extracted_watermarks')) #0.0079
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'extracted_watermarks_NEW')) #0.0513 Max BER: 0.5044
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'extracted_watermarks')) # 0.2249
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20','watermarked_lab','jpeg50', 'extracted')) #0.4343
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab', 'jpeg80', 'extracted')) #0.3010
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab', 'jpeg80', 'extracted'))  # 0.3010
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'restormer_extracted')) # avg 0.4535 max 0.51 min 0.3468
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab_NEW', 'jpeg50', 'extracted')) # avg 0.4360 max 0.5194 min 0.2505
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'dlnetEncoder32_9_40_alpha20', 'watermarked_lab_NEW', 'jpeg50', 'extracted')) #dir_001  avg: 0.4378 Max: 0.5081 Min: 0.3358
    # evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'restormer_extracted')) #new_extractor avg: 0.4536 Max: 0.5137 Min: 0.3390
    evaluate_cwv_accuracy(os.path.join('..', 'FullNS', 'mmif_cddfuse_extracted')) #dir_001 avg: 0.4401 Max: 0.5056 Min: 0.3360