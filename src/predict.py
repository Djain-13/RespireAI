import torch
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import get_model
from gradcam import GradCAM, overlay_heatmap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# FIX: Load per-disease thresholds saved by evaluate.py
# Falls back to 0.5 for all if the file doesn't exist yet
try:
    with open("../best_thresholds.json") as f:
        threshold_dict = json.load(f)
    thresholds = [threshold_dict[l] for l in labels]
    print("Loaded per-disease thresholds from best_thresholds.json")
except FileNotFoundError:
    thresholds = [0.5] * 14
    print("best_thresholds.json not found — using default threshold 0.5 for all diseases")


def create_lung_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask = mask / 255.0
    return mask


model = get_model()
model.load_state_dict(torch.load("../best_model.pth", map_location=device))
model.to(device)
model.eval()

target_layer = model.features[-1]
gradcam = GradCAM(model, target_layer)

image_path = input("Enter X-ray image path: ")
image = Image.open(image_path).convert("RGB")
img_array = np.array(image.resize((224, 224)))
input_tensor = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    output = model(input_tensor)
    probs = torch.sigmoid(output).cpu().numpy()[0]


print("\n========== Predicted Disease Probabilities ==========\n")
detected = []
for i, p in enumerate(probs):
    marker = "  *** DETECTED" if p > thresholds[i] else ""
    print(f"{labels[i]:<22}: {p:.3f}{marker}")
    if p > thresholds[i]:
        detected.append((labels[i], p))

if not detected:
    print("\nNo diseases detected above threshold.")
else:
    print(f"\nDetected: {', '.join([f'{d} ({p:.2f})' for d, p in detected])}")


# Grad-CAM on the highest-probability class
target_class = int(np.argmax(probs))
print(f"\nGenerating Grad-CAM for: {labels[target_class]} (p={probs[target_class]:.3f})")

cam = gradcam.generate(input_tensor, target_class)

mask = create_lung_mask(img_array)
cam = cam * mask

threshold_cam = 0.6
affected_pixels = (cam > threshold_cam).sum()
total_pixels = cam.size
damage_percent = (affected_pixels / total_pixels) * 100

if damage_percent < 10:
    severity = "Mild"
elif damage_percent < 30:
    severity = "Moderate"
elif damage_percent < 60:
    severity = "Severe"
else:
    severity = "Critical"

print("\n========== Lung Damage Analysis ==========")
print(f"Affected Lung Area : {damage_percent:.2f}%")
print(f"Severity Level     : {severity}")

overlay = overlay_heatmap(img_array, cam)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original X-ray")
plt.imshow(img_array)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Grad-CAM — {labels[target_class]}")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()