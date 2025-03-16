from model import PneumoniaClassifier  # Reuse the same model class
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

# Load test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_dataset = datasets.ImageFolder('data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PneumoniaClassifier().to(device)

model.load_state_dict(torch.load('pneumonia_classifier.pth'))
model.eval()

# Evaluate on test set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
