import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 2️⃣ Data Preprocessing & Augmentation
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


train_dir = r"C:\Users\HP\Desktop\add_models\Pneumonia\train"
val_dir = r"C:\Users\HP\Desktop\add_models\Pneumonia\validate"

# Load train & validation datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders
batch_size = 16  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


print(f"Classes: {train_dataset.classes}")

# ------------------------------
# 3️⃣ Load Pretrained DenseNet169
# ------------------------------
from torchvision.models import densenet169, DenseNet169_Weights


model = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)


num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)  

model = model.to(device)


criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ------------------------------
# 5️⃣ Training Function with Early Stopping
# ------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30, patience=5):
    best_val_loss = np.inf
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Reduce LR if no improvement
        scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "pneumonia_densenet169.pth")  # Save best model
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered!")
            break

# ------------------------------
# 6️⃣ Validation Function
# ------------------------------
def evaluate_model(model, val_loader, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    return val_loss / len(val_loader), val_acc

# ------------------------------
# 7️⃣ Train the Model
# ------------------------------
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30, patience=5)

# ------------------------------
# 8️⃣ Load & Test the Best Model
# ------------------------------
model.load_state_dict(torch.load("pneumonia_densenet169.pth", weights_only=True))
model.eval()
print("Best model loaded and ready for testing!")
