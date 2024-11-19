import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class DeepFakeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = sorted([os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Label assignment based on filename
        label = 1.0 if "real" in img_path.lower() else 0.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Define the CFFN Module
class CFFN(nn.Module):
    def __init__(self, in_channels):
        super(CFFN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# Define the DeepFakeDetector Model
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.cffn = CFFN(in_channels=1024)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        features = self.densenet.features(x)
        fused_features = self.cffn(features)
        pooled = nn.AdaptiveAvgPool2d((1, 1))(fused_features)
        flattened = pooled.view(pooled.size(0), -1)
        output = self.fc(flattened)
        return output

# Define the Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
model = DeepFakeDetector().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Prepare the Dataloader
train_real_dir = "data/Dataset/Train/Real"
train_fake_dir = "data/Dataset/Train/Fake"
real_dataset = DeepFakeDataset(train_real_dir, transform)
fake_dataset = DeepFakeDataset(train_fake_dir, transform)
train_dataset = real_dataset + fake_dataset
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Training Function
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "deepfake_detector.pth")
    print("Model saved as 'deepfake_detector.pth'")

if __name__ == "__main__":
    train()
