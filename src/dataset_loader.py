import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_root, transform=None):

        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform

        self.labels = [
            'Atelectasis','Cardiomegaly','Effusion','Infiltration',
            'Mass','Nodule','Pneumonia','Pneumothorax',
            'Consolidation','Edema','Emphysema','Fibrosis',
            'Pleural_Thickening','Hernia'
        ]
        self.image_paths = {}

        for root, dirs, files in os.walk(image_root):
            for file in files:
                if file.endswith(".png"):
                    self.image_paths[file] = os.path.join(root, file)

        print("Total images found:", len(self.image_paths))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        img_name = row["Image Index"]
        label_string = row["Finding Labels"]

        img_path = self.image_paths[img_name]

        image = Image.open(img_path).convert("RGB")

        label_vector = [0]*14

        if label_string != "No Finding":

            diseases = label_string.split("|")

            for disease in diseases:
                if disease in self.labels:
                    label_vector[self.labels.index(disease)] = 1

        if self.transform:
            image = self.transform(image)

        import torch
        return image, torch.tensor(label_vector)