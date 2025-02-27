import json
import os
import numpy as np
from PIL import Image
from pycocotools.mask import decode
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2


class ToolDataset(Dataset):
    def __init__(self, annotation_file, image_dir, image_transform=None, mask_transform=None):
        if not os.access(annotation_file, os.R_OK):
            raise PermissionError(f"Нет прав на чтение файла аннотации: {annotation_file}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Директория с изображениями не найдена: {image_dir}")

        with open(annotation_file, 'r') as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.images = [img['file_name'] for img in self.coco['images']]
        self.annotations = self.coco['annotations']

        self.img_to_anns = {}
        for ann in self.annotations:
            if ann['image_id'] not in self.img_to_anns:
                self.img_to_anns[ann['image_id']] = []
            self.img_to_anns[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        img_info = self.coco['images'][idx]
        img_id = img_info['id']
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')

        mask = np.zeros((img_info['height'], img_info['width'], 4), dtype=np.uint8)  # 4-канальная маска

        anns = self.img_to_anns.get(img_id, [])
        for ann in anns:
            if 'segmentation' in ann:
                category_id = ann['category_id']  # Класс маски (1, 2, 3, 4)
                if isinstance(ann['segmentation'], list):
                    for poly in ann['segmentation']:
                        poly = np.array(poly).reshape(-1, 1, 2).astype(np.int32)
                        cv2.fillPoly(np.ascontiguousarray(mask[..., category_id - 1]), [poly], color=1)

                else:
                    rle = ann['segmentation']
                    decoded_mask = decode(rle)
                    mask[..., category_id - 1] = np.maximum(mask[..., category_id - 1], decoded_mask)

        mask = Image.fromarray(mask)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.images)
