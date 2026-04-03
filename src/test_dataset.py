from dataset_loader import ChestXrayDataset

dataset = ChestXrayDataset(
    csv_file="../archive/Data_Entry_2017.csv",
    image_root="../archive"
)

print("Total images:", len(dataset))
img, label = dataset[0]
print("Sample label:", label)