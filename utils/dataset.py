import os

import torch.utils.data as Data
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import transforms
from torchvision.transforms import functional as F

class TopLeftCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        return F.crop(img, top=0, left=0, height=self.height, width=self.width)

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR)

class WatermarkDataset(Data.Dataset):
    def __init__(self, root_dir, transform=None,num_dirs=20,num_igm=50):
        """
        root_dir — путь до 'dlnetEncoder32_9_40_alpha20'
        transform — torchvision.transforms, например ToTensor()
        """
        self.root_dir = root_dir
        self.transform = transform if transform else T.ToTensor()
        self.transform = transform or transforms.Compose([
            TopLeftCrop(224,480),  # обрезка по центру до 1080x1920
            transforms.ToTensor()
        ])

        self.watermarked_dir = os.path.join(root_dir, "watermarked")
        self.compressed_dir = os.path.join(root_dir, "watermarked_lab", "jpeg50", "attacked_images")

        self.pairs = []

        # обходим все dir_XXX
        for dir_name in sorted(os.listdir(self.watermarked_dir))[2:num_dirs]:
            wm_path = os.path.join(self.watermarked_dir, dir_name)
            comp_path = os.path.join(self.compressed_dir, dir_name)

            if not os.path.isdir(wm_path) or not os.path.isdir(comp_path):
                continue

            wm_images = sorted(os.listdir(wm_path))[:num_igm]
            comp_images = sorted(os.listdir(comp_path))[:num_igm]

            for wm_img, comp_img in zip(wm_images, comp_images):
                self.pairs.append((
                    os.path.join(comp_path, comp_img),  # input JPEG image
                    os.path.join(wm_path, wm_img),      # target image with ЦВЗ
                    comp_img# filename
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        comp_path, wm_path, name = self.pairs[idx]

        comp_img = Image.open(comp_path).convert('RGB')
        wm_img = Image.open(wm_path).convert('RGB')

        comp_tensor = self.transform(comp_img)
        wm_tensor = self.transform(wm_img)
        return comp_tensor, wm_tensor, name

if __name__ == "__main__":
    dataset = WatermarkDataset(
        root_dir=r"C:\Users\user\deep-image-prior\data\FullNS\dlnetEncoder32_9_40_alpha20"
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    i = 1
    d = {}
    for x_input, x_target, name in loader:
        d[(x_input.shape[2],x_input.shape[3])] = d.get((x_input.shape[2],x_input.shape[3]),0)+1
        # print("JPEG:", x_input.shape, "ORIGINAL:", x_target.shape, "FILENAME:", name[0], i)
    print(d)