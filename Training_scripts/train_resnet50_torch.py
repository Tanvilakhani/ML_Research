import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === CONFIG ===
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
FINE_TUNE_EPOCHS = 5
NUM_CLASSES = 10
TRAIN_DIR = r'C:\Users\admin\Documents\ML_research\MY_data\train'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PATHS ===
SCRIPTED_PATH = 'models/resnet50_fruits_scripted.pt'
CSV_PATH = 'results/report/resnet50_training_history(torch).csv'
REPORT_PATH = 'results/report/resnet50_classification_report(torch).txt'
METRICS_PLOT_PATH = 'results/metrics/resnet50_training_metrics(torch).png'
CONF_MATRIX_PATH = 'results/metrics/resnet50_confusion_matrix(torch).png'
os.makedirs(os.path.dirname(SCRIPTED_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PLOT_PATH), exist_ok=True)

# === DATASET ===
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(TRAIN_DIR)
class_names = dataset.classes
num_samples = len(dataset)
split = int(0.8 * num_samples)
train_set, val_set = torch.utils.data.random_split(dataset, [split, num_samples - split])
train_set.dataset.transform = transform_train
val_set.dataset.transform = transform_val

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# === MODEL ===
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)
model = model.to(DEVICE)

# === TRAINING FUNCTION ===
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_loss /= total
        train_acc = correct / total

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        val_loss /= total
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
    return history

# === TRAIN ===
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
history1 = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# === FINE-TUNING ===
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
history2 = train_model(model, train_loader, val_loader, criterion, optimizer, FINE_TUNE_EPOCHS)

# === SAVE TORCHSCRIPT ONLY ===
model.eval()
example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save(SCRIPTED_PATH)

# === EVALUATION ===
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open(REPORT_PATH, 'w') as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()

# === SAVE HISTORY ===
epochs_total = list(range(1, EPOCHS + FINE_TUNE_EPOCHS + 1))
history = pd.DataFrame({
    'epoch': epochs_total,
    'train_loss': history1['train_loss'] + history2['train_loss'],
    'val_loss': history1['val_loss'] + history2['val_loss'],
    'train_acc': history1['train_acc'] + history2['train_acc'],
    'val_acc': history1['val_acc'] + history2['val_acc']
})
history.to_csv(CSV_PATH, index=False)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['train_acc'], label='Train Accuracy')
plt.plot(history['epoch'], history['val_acc'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.tight_layout()
plt.savefig(METRICS_PLOT_PATH)
plt.close()
