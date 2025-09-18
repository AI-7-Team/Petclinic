import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import LesionDataset
from model import build_resnet50

# ---------------------
# 하이퍼파라미터
# ---------------------
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
VAL_RATIO = 0.2
DATA_DIRS = [
   #aaa
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ---------------------
    # Dataset / Split
    # ---------------------
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    full_dataset = LesionDataset(DATA_DIRS, transform=tfm)

    num_val = int(len(full_dataset) * VAL_RATIO)
    num_train = len(full_dataset) - num_val
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ---------------------
    # Model / Loss / Optim
    # ---------------------
    model = build_resnet50(num_classes=len(full_dataset.class_to_idx))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------------------
    # Training Loop
    # ---------------------
    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        train_loss, train_correct = 0.0, 0

        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        train_loss /= num_train
        train_acc = train_correct / num_train

        # ---- validate ----
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}/{EPOCHS}"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss /= num_val
        val_acc = val_correct / num_val

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # ---------------------
    # Save model
    # ---------------------
    torch.save(model.state_dict(), "resnet50_lesion.pth")
    print("✅ Training complete")

if __name__ == '__main__':
    main()
