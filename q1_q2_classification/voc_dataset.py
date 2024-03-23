import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(Dataset):
    # classes = ['0_black_3_legs', '5_NPN_Bipolar_Transistors', '6_Black_Rec']
    classes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    def __init__(self, split, size, data_dir):
        self.data_dir = data_dir
        self.size = size
        self.split = split
        self.classes = VOCDataset.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()
        # print(self.class_to_idx)

    def _load_images(self):
        images = []
        data_split_path = "train" if self.split == "train" else "test"
        for cls in self.classes:
            class_path = os.path.join(self.data_dir, data_split_path, cls)
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"Directory not found: {class_path}")

            class_images = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]

            for img_path in class_images:
                # Check if img_path ends with a valid image extension
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    images.append((img_path, self.class_to_idx[cls]))
                else:
                    print(f"Invalid image file: {img_path}")
        return images


    def __len__(self):
        return len(self.images)

    def get_random_augmentations(self):
        # Define a list of possible augmentations
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.CenterCrop(size=(self.size, self.size)),
                transforms.RandomRotation(degrees=(-45, 45)),
            ]
        else:
            augmentations = [
                # transforms.CenterCrop(size=(self.size, self.size))
            ]
        return augmentations
        
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        trans = transforms.Compose([
            transforms.Resize(self.size),
            *self.get_random_augmentations(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.457, 0.407], std=[0.5, 0.5, 0.5]),
        ])
 
        image = trans(image)
        weight = torch.ones(len(self.classes))
        class_vec = torch.zeros(len(self.classes))
        class_vec[label] = 1
        return image, class_vec, weight
