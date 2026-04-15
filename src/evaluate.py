import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import os
from PIL import Image

DATA_CSV  = "../archive/Data_Entry_2017.csv"
DATA_ROOT = "../archive"
MODEL_PATH = "../best_model.pth"

LABELS = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pneumonia','Pneumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.image_paths = {}
        for root, dirs, files in os.walk(image_root):
            for f in files:
                if f.endswith(".png"):
                    self.image_paths[f] = os.path.join(root, f)
        print(f"Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["Image Index"]
        label_vector = [0] * 14
        if row["Finding Labels"] != "No Finding":
            for d in row["Finding Labels"].split("|"):
                if d in LABELS:
                    label_vector[LABELS.index(d)] = 1
        if img_name not in self.image_paths:
            return self.__getitem__((idx + 1) % len(self.data))
        try:
            image = Image.open(self.image_paths[img_name]).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_vector)

def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ChestXrayDataset(DATA_CSV, DATA_ROOT, val_transform)
    test_indices = list(range(len(dataset) - 10000, len(dataset)))
    test_dataset = Subset(dataset, test_indices)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    print(f"Test set size: {len(test_dataset)}")

    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(lbls.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    best_thresholds = []
    print("\nPer-disease results:")
    print(f"{'Disease':<22} {'Threshold':>10} {'AUC':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 60)

    for i, disease in enumerate(LABELS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except Exception:
            auc = 0.0

        prec, rec, thresholds = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        f1s = (2 * prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        best_t   = float(thresholds[min(best_idx, len(thresholds)-1)])
        best_thresholds.append(best_t)

        preds_i = (all_preds[:, i] > best_t).astype(int)
        tp = int(((preds_i == 1) & (all_labels[:, i] == 1)).sum())
        fp = int(((preds_i == 1) & (all_labels[:, i] == 0)).sum())
        fn = int(((preds_i == 0) & (all_labels[:, i] == 1)).sum())

        print(f"{disease:<22} {best_t:>10.3f} {auc:>7.3f} {tp:>5} {fp:>5} {fn:>5}")

    pred_binary = np.zeros_like(all_preds)
    for i in range(len(best_thresholds)):
        pred_binary[:, i] = (all_preds[:, i] > best_thresholds[i]).astype(int)

    precision = precision_score(all_labels, pred_binary, average="macro", zero_division=0)
    recall    = recall_score(all_labels, pred_binary, average="macro", zero_division=0)
    f1        = f1_score(all_labels, pred_binary, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_preds, average="macro")
    except Exception:
        auc = 0.0

    print("\n========== Overall Results ==========")
    print(f"Precision (macro) : {precision:.3f}")
    print(f"Recall    (macro) : {recall:.3f}")
    print(f"F1        (macro) : {f1:.3f}")
    print(f"ROC-AUC   (macro) : {auc:.3f}")

    # Save thresholds
    import json
    threshold_dict = {LABELS[i]: best_thresholds[i] for i in range(len(LABELS))}
    with open("../best_thresholds.json", "w") as f:
        json.dump(threshold_dict, f, indent=2)
    print("\nThresholds saved to best_thresholds.json")

if __name__ == "__main__":
    main()