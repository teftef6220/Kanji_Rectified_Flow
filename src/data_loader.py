import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class KanjiDataLoader(Dataset):
    def __init__(self, data_dir, json_path,args, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.args = args
        with open(json_path, "r", encoding="utf-8") as f:
            self.data_dict = json.load(f)
        self.file_list = list(self.data_dict.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.data_dir, file_name)
        image = Image.open(image_path).convert("L")  # RGBに変換
        if self.args.use_label_clip_embedding:
            label = torch.tensor(self.data_dict[file_name]["clip_embedding"], dtype=torch.float)
        else:
            label = torch.tensor(self.data_dict[file_name]["label"], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, label
