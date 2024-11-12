import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 64
image_size = 64
lr = 0.0001
num_epochs = 30

class DCGANDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        features = self.main[:-1](x)  
        return features.view(features.size(0), -1)

class PairwiseModel(nn.Module):
    def __init__(self, feature_extractor):
        super(PairwiseModel, self).__init__()
        self.feature_extractor = feature_extractor
        
    
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 64, 64).to(next(feature_extractor.parameters()).device)
            sample_output = feature_extractor(sample_input)
            feature_size = sample_output.shape[1]
            
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        print(f"Initialized classifier with input size: {feature_size * 2}")

    def forward(self, real_images, fake_images):
        
        with torch.no_grad():  
            real_features = self.feature_extractor(real_images)
            fake_features = self.feature_extractor(fake_images)
        
        
        combined_features = torch.cat((real_features, fake_features), dim=1)
        
        
        return self.classifier(combined_features)

class PairwiseDataset(Dataset):
    def __init__(self, real_folder, fake_folder, transform=None):
        self.real_folder = Path(real_folder)
        self.fake_folder = Path(fake_folder)
        self.transform = transform
        
        self.real_images = list(self.real_folder.glob('*.jpg')) + list(self.real_folder.glob('*.png'))
        self.fake_images = list(self.fake_folder.glob('*.jpg')) + list(self.fake_folder.glob('*.png'))
        
        
        if not self.real_images or not self.fake_images:
            raise ValueError("No images found in one or both folders")
            
        
        self.dataset_size = min(len(self.real_images), len(self.fake_images))
        
        print(f"Found {len(self.real_images)} real images and {len(self.fake_images)} fake images")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        try:
            
            real_img_path = self.real_images[idx % len(self.real_images)]
            fake_img_path = self.fake_images[idx % len(self.fake_images)]
            
            
            real_img = cv2.imread(str(real_img_path))
            fake_img = cv2.imread(str(fake_img_path))
            
            if real_img is None or fake_img is None:
                raise ValueError(f"Failed to load images: {real_img_path} or {fake_img_path}")
            
            
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
            
            
            if self.transform:
                real_img = self.transform(real_img)
                fake_img = self.transform(fake_img)
                
            return real_img, fake_img, 1.0, 0.0  
            
        except Exception as e:
            print(f"Error loading images at index {idx}: {str(e)}")
            
            return self.__getitem__(random.randint(0, len(self) - 1))

def verify_model_architecture(model):
    """
    Print model architecture and verify its structure
    """
    print("\nModel Architecture:")
    print(model)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    
    sample_input = torch.randn(1, 3, 64, 64).to(device)  
    try:
        with torch.no_grad():
            features = model(sample_input)
        print(f"Feature output shape: {features.shape}")
        return True
    except Exception as e:
        print(f"Error in model architecture: {str(e)}")
        return False

def load_model_safely(model, model_path):
    """
    Safely load the model weights with proper error handling
    """
    try:
        
        if device.type == 'cuda':
            state_dict = torch.load(str(model_path), weights_only=True)
        else:
            state_dict = torch.load(str(model_path), weights_only=True, map_location=torch.device('cpu'))
        
        
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        unexpected = state_dict_keys - model_keys
        missing = model_keys - state_dict_keys
        
        if unexpected:
            print(f"Warning: Unexpected keys in state dict: {unexpected}")
        if missing:
            print(f"Warning: Missing keys in state dict: {missing}")
            
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def main():
    try:
        
        ngpu = 1
        feature_extractor = DCGANDiscriminator(ngpu).to(device)
        
        
        print("\nVerifying model architecture...")
        if not verify_model_architecture(feature_extractor):
            raise ValueError("Model architecture verification failed")
        
        
        model_path = Path("model/dcgan_discriminator.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Pre-trained model not found at: {model_path}")
        
        if not load_model_safely(feature_extractor, model_path):
            raise ValueError("Failed to load pre-trained model")
        
        feature_extractor.eval()
        
        
        model = PairwiseModel(feature_extractor).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        real_images_folder = Path("data/Dataset/Train/Real")
        fake_images_folder = Path("data/Dataset/Train/Fake")
        
        
        print(f"\nDataset paths:")
        print(f"Real images: {real_images_folder.absolute()}")
        print(f"Fake images: {fake_images_folder.absolute()}")
        
        dataset = PairwiseDataset(
            real_folder=real_images_folder,
            fake_folder=fake_images_folder,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"\nDataset size: {len(dataset)} pairs")
        print(f"Number of batches: {len(dataloader)}")
        
        
        print("\nStarting Training Loop...")
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (real_images, fake_images, real_labels, fake_labels) in enumerate(dataloader):
                try:
                    
                    real_images = real_images.to(device)
                    fake_images = fake_images.to(device)
                    real_labels = torch.ones(real_images.size(0), 1).to(device)
                    fake_labels = torch.zeros(fake_images.size(0), 1).to(device)

                    optimizer.zero_grad()
                    outputs = model(real_images, fake_images)
                    labels = torch.cat((real_labels, fake_labels))
                    predictions = torch.cat((outputs, -outputs))

                    loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    if i % 50 == 0:
                        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                              f"Loss: {loss.item():.4f}, "
                              f"Device: {device}")
                
                except Exception as e:
                    print(f"Error in training batch: {str(e)}")
                    continue
            
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        
        save_path = Path('model')
        save_path.mkdir(exist_ok=True)
        torch.save(
            model.state_dict(),
            save_path / 'pairwise_detector_with_dcgan.pth',
            _use_new_zipfile_serialization=True
        )
        print("Training complete. Model saved!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()