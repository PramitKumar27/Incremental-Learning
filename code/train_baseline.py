import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# --- REPRODUCIBILITY ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

g = torch.Generator().manual_seed(SEED)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=g)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

print("Starting Training...")
epochs = 1
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50:.3f}')
            running_loss = 0.0

# Save checkpoint (with metadata)
torch.save({
    "model_name": "vit_tiny_patch16_224",
    "num_classes": 10,
    "seed": SEED,
    "state_dict": model.state_dict(),
}, "./vit_eurosat_clean.pth")
print("Saved clean checkpoint.")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
