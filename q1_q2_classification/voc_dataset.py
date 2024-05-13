import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import functional

class CustomTransform(torch.nn.Module):
    def forward(self, img):  # we assume inputs are always structured like this
        color = img.getpixel((0, 0))
        w, h = img.size
        trans = transforms.Pad(padding=(1080 - w, 720 - h), padding_mode="constant", fill=color)
        img = trans(img)
        return img

class VOCDataset(Dataset):
    # classes = ['0_black_3_legs', '5_NPN_Bipolar_Transistors', '6_Black_Rec']
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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
        if self.split == "train":
            data_split_path = "train" 
        elif self.split == "val":
            data_split_path = "val" 
        elif self.split == "test":
            data_split_path = "test" 
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
            
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

    def get_random_augmentations_1(self):
        # Define a list of possible augmentations
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        
        return augmentations

    def get_random_augmentations_2(self):
        # Define a list of possible augmentations
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-45, 45), fill = 255),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),

            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        return augmentations

    def get_random_augmentations_3(self):
        # Define a list of possible augmentations
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=(-45, 45), fill = 255),
            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        return augmentations


    def get_random_augmentations_4(self):
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(degrees=(-45, 45), fill = 255),
            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        return augmentations

    def get_random_augmentations_5(self):
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=(-45, 45), translate=(0.3, 0.3), fill = 255),
            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        return augmentations
    
    def get_random_augmentations_6(self):
        if self.split == "train":
            augmentations = [
                transforms.RandomCrop((1080, 1080)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomAffine(degrees=(-180, 180), translate=(0.3, 0.3), fill = 255),
            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.CenterCrop((1080, 1080)),
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        return augmentations
        
    def get_random_augmentations_7(self):
        if self.split == "train":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        elif self.split == "val" or self.split == "test":
            augmentations = [
                transforms.RandomHorizontalFlip(p=0),
            ]
        else:
            raise ValueError("Invalid value for 'split'. Expected 'train', 'val' or 'test'.")
        return augmentations
            
        
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        trans = transforms.Compose([
            *self.get_random_augmentations_7(),
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
 
        image = trans(image)
        weight = torch.ones(len(self.classes))
        class_vec = torch.zeros(len(self.classes))
        class_vec[label] = 1
        return image, class_vec, weight
