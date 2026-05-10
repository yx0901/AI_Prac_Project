import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_default_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def build_training_transform():
    return transforms.Compose([
        transforms.RandomRotation(10, fill=255),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5,
            fill=255,
        ),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3, fill=255),
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

class ChineseCharDataset(Dataset):
    def __init__(self, root_dir='data/train', transform=None):
        self.root_dir = root_dir
        self.transform = transform or build_default_transform()
        
        # Find all the chracter classes
        all_items = sorted(os.listdir(root_dir))
        folder_names = []
        for item in all_items:
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path): #If it is a directory (folder), it is a character class
                folder_names.append(item)
        self.classes = sorted(folder_names)

        # Create mappings: character to number
        self.class_to_idx = {}  # {'一': 0, '七': 1, ...}
        self.idx_to_class = {}  # {0: '一', 1: '七', ...}

        for i, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = i
            self.idx_to_class[i] = class_name
        
        # Collect all image paths and labels
        self.samples = []
        for character in self.classes:
            folder_path = os.path.join(root_dir, character)
            label = self.class_to_idx[character]
            for img_name in sorted(os.listdir(folder_path)):
                if img_name.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label