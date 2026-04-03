import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler, Dataset
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import random
import os
from PIL import Image

DATA_CSV  = "../archive/Data_Entry_2017.csv"
DATA_ROOT = "../archive"
MODEL_OUT = "../best_model.pth"


LABELS = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pneumonia','Pneumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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
        print(f"Total images found: {len(self.image_paths)}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(self.image_paths[row["Image Index"]]).convert("RGB")
        label_vector = [0] * 14
        if row["Finding Labels"] != "No Finding":
            for d in row["Finding Labels"].split("|"):
                if d in LABELS:
                    label_vector[LABELS.index(d)] = 1
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_vector)


def get_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    return model


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    full_dataset = ChestXrayDataset(DATA_CSV, DATA_ROOT, train_transform)

    SUBSET_SIZE = 15000
    indices = random.sample(range(len(full_dataset)), SUBSET_SIZE)
    subset  = Subset(full_dataset, indices)

    TRAIN_SIZE = 12750
    VAL_SIZE   = 2250
    train_ds, val_ds = random_split(subset, [TRAIN_SIZE, VAL_SIZE])

    df = pd.read_csv(DATA_CSV).iloc[indices].reset_index(drop=True)
    dm = df["Finding Labels"].str.get_dummies(sep="|")
    if "No Finding" in dm.columns:
        dm = dm.drop(columns=["No Finding"])
    for label in LABELS:
        if label not in dm.columns:
            dm[label] = 0
    dm = dm[LABELS]

    class_counts = dm.sum().replace(0, 1)
    pos_weight = torch.clamp(
        torch.tensor(
            ((len(dm) - class_counts) / class_counts).to_numpy(),
            dtype=torch.float32
        ),
        min=1.0, max=10.0
    ).to(device)

    sw = dm.dot(1.0 / class_counts).values.copy()
    train_weights = torch.tensor(sw[train_ds.indices], dtype=torch.float32)

    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False,   num_workers=0)

    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val   batches: {len(val_loader)}",   flush=True)

    model = get_model()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val   = float("inf")
    no_improve = 0
    phase      = 1
    EPOCHS     = 15

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}  [Phase {phase}]", flush=True)

        if epoch == 4 and phase == 1:
            phase = 2
            print(">>> Phase 2: unfreezing denseblock4 + norm5 <<<", flush=True)
            for n, p in model.features.named_parameters():
                if "denseblock4" in n or "norm5" in n:
                    p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": [p for n, p in model.features.named_parameters()
                            if "denseblock4" in n or "norm5" in n], "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

        if epoch == 9 and phase == 2:
            phase = 3
            print(">>> Phase 3: unfreezing full backbone <<<", flush=True)
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.features.parameters(),   "lr": 2e-6},
                {"params": model.classifier.parameters(), "lr": 2e-5},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

        model.train()
        tloss = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Train Ep{epoch+1}"):
            imgs = imgs.to(device)
            lbls = lbls.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tloss += loss.item()
        print(f"Train Loss: {tloss/len(train_loader):.4f}", flush=True)

        model.eval()
        vloss = 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Val Ep{epoch+1}"):
                imgs = imgs.to(device)
                lbls = lbls.float().to(device)
                vloss += criterion(model(imgs), lbls).item()
        avg_val = vloss / len(val_loader)
        print(f"Val   Loss: {avg_val:.4f}", flush=True)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val   = avg_val
            no_improve = 0
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"Model saved! (best val loss: {best_val:.4f})", flush=True)
        else:
            no_improve += 1
            print(f"No improvement {no_improve}/5", flush=True)
            if no_improve >= 5:
                print("Early stopping triggered.", flush=True)
                break

    print(f"\nTraining complete!", flush=True)
    print(f"Best val loss : {best_val:.4f}", flush=True)
    print(f"Model saved at: {MODEL_OUT}", flush=True)

if __name__ == "__main__":
    main()