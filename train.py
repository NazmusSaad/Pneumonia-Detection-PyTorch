import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from model import PneumoniaClassifier  # Import from model.py

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PneumoniaClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    running_loss = 0.0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/10 Loss: {running_loss/len(train_loader):.4f}")

# Save Model Weights (only weights now, not whole object)
torch.save(model.state_dict(), 'pneumonia_classifier.pth')
print("Model weights saved as 'pneumonia_classifier.pth'.")

# Save Labels
with open('labels.txt', 'w') as f:
    for idx, class_name in enumerate(train_dataset.classes):
        f.write(f"{idx} {class_name}\n")

print("Labels saved to 'labels.txt'.")
