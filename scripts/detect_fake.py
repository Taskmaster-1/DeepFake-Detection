import os
import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Modified hyperparameters
batch_size = 32  # Reduced batch size for better stability
image_size = 64
lr = 0.0003  # Slightly increased learning rate
num_epochs = 30

def load_model_safely(model, model_path):
    """
    Safely load the model weights with proper error handling
    Args:
        model: PyTorch model to load weights into
        model_path: Path to the saved model weights
    Returns:
        bool: True if loading successful, False otherwise
    """
    try:
        # Load state dict based on device type
        if device.type == 'cuda':
            state_dict = torch.load(str(model_path))
        else:
            state_dict = torch.load(str(model_path), map_location=torch.device('cpu'))
        
        # Check for key mismatches
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        unexpected = state_dict_keys - model_keys
        missing = model_keys - state_dict_keys
        
        if unexpected:
            print(f"Warning: Unexpected keys in state dict: {unexpected}")
        if missing:
            print(f"Warning: Missing keys in state dict: {missing}")
        
        # Try to load the state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded successfully from {model_path}")
        return True
    
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return False
    except RuntimeError as e:
        print(f"Error loading model: {str(e)}")
        print("This might be due to model architecture mismatch or corrupted file")
        return False
    except Exception as e:
        print(f"Unexpected error loading model: {str(e)}")
        return False

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
        
        # Keep feature extractor trainable for fine-tuning
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 64, 64).to(next(feature_extractor.parameters()).device)
            sample_output = feature_extractor(sample_input)
            feature_size = sample_output.shape[1]
            
        # Modified classifier architecture with batch normalization and different dropout rates
        self.classifier = nn.Sequential(
            nn.Linear(feature_size * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        print(f"Initialized classifier with input size: {feature_size * 2}")

    def forward(self, real_images, fake_images):
        # Remove torch.no_grad() to allow feature extractor training
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
            fake_img_path = self.fake_images[random.randint(0, len(self.fake_images) - 1)]  # Random pairing
            
            real_img = cv2.imread(str(real_img_path))
            fake_img = cv2.imread(str(fake_img_path))
            
            if real_img is None or fake_img is None:
                raise ValueError(f"Failed to load images: {real_img_path} or {fake_img_path}")
            
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                real_img = self.transform(real_img)
                fake_img = self.transform(fake_img)
                
            return real_img, fake_img, torch.tensor(1.0), torch.tensor(0.0)
            
        except Exception as e:
            print(f"Error loading images at index {idx}: {str(e)}")
            return self.__getitem__(random.randint(0, len(self) - 1))

def find_learning_rate(model, train_loader, criterion, device, start_lr=1e-7, end_lr=1, num_iter=100):
    """
    Implements the learning rate range test described in the paper 
    "Cyclical Learning Rates for Training Neural Networks"
    """
    log_lrs = torch.linspace(math.log(start_lr), math.log(end_lr), num_iter)
    learning_rates = torch.exp(log_lrs)
    
    # Initialize optimizer with minimum learning rate
    optimizer = optim.AdamW(model.parameters(), lr=start_lr)
    
    losses = []
    best_loss = float('inf')
    
    model.train()
    for i, lr in enumerate(learning_rates):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr.item()
            
        # Get batch
        try:
            real_images, fake_images, real_labels, fake_labels = next(iter(train_loader))
        except StopIteration:
            train_loader_iter = iter(train_loader)
            real_images, fake_images, real_labels, fake_labels = next(train_loader_iter)
            
        real_images = real_images.to(device)
        fake_images = fake_images.to(device)
        real_labels = torch.ones(real_images.size(0), 1).to(device) * 0.9
        fake_labels = torch.zeros(fake_images.size(0), 1).to(device) + 0.1
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(real_images, fake_images)
        
        # Calculate loss
        loss = criterion(outputs, real_labels)
        loss += criterion(-outputs, fake_labels)
        loss = loss / 2
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Stop if loss explodes
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss:
            break
            
        if i % 10 == 0:
            print(f'Learning rate: {lr:.7f}, Loss: {loss.item():.4f}')
    
    return learning_rates[:len(losses)], losses

def plot_learning_rate(lrs, losses):
    """Plot the learning rate range test results"""
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.grid(True)
    plt.savefig('learning_rate_finder.png')
    plt.close()

def main():
    try:
        # 1. Set up model
        ngpu = 1
        feature_extractor = DCGANDiscriminator(ngpu).to(device)
        
        # 2. Load pre-trained weights if available
        model_path = Path("model/dcgan_discriminator.pth")
        if model_path.exists():
            if not load_model_safely(feature_extractor, model_path):
                print("Warning: Could not load pre-trained model. Starting with randomly initialized weights.")
        else:
            print("No pre-trained model found. Starting with randomly initialized weights.")
        
        model = PairwiseModel(feature_extractor).to(device)
        
        # 3. Set up dataset and dataloader
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        real_images_folder = Path("data/Dataset/Train/Real")
        fake_images_folder = Path("data/Dataset/Train/Fake")
        
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
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True
        )
        
        # 4. Set up loss function
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 5. Find optimal learning rate
        print("Finding optimal learning rate...")
        learning_rates, losses = find_learning_rate(
            model=model,
            train_loader=dataloader,
            criterion=criterion,
            device=device
        )

        plot_learning_rate(learning_rates, losses)

        # Choose learning rate at minimum loss or order of magnitude less than where loss starts to climb
        min_loss_idx = np.argmin(losses)
        optimal_lr = learning_rates[min_loss_idx].item() / 10  # Divide by 10 for stability

        print(f"Optimal learning rate found: {optimal_lr:.7f}")

        # Initialize optimizer with found learning rate
        optimizer = optim.AdamW(model.parameters(), lr=optimal_lr, weight_decay=0.01)
        
        # Set up learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        print("\nStarting Training Loop...")
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()  # Ensure model is in training mode
            epoch_loss = 0.0
            batch_count = 0
            
            for i, (real_images, fake_images, real_labels, fake_labels) in enumerate(dataloader):
                try:
                    real_images = real_images.to(device)
                    fake_images = fake_images.to(device)
                    
                    # Label smoothing
                    real_labels = torch.ones(real_images.size(0), 1).to(device) * 0.9  # Smooth positive labels
                    fake_labels = torch.zeros(fake_images.size(0), 1).to(device) + 0.1  # Smooth negative labels

                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(real_images, fake_images)
                    
                    # Calculate loss with proper reshaping
                    loss = criterion(outputs, real_labels)  # Real pairs should output 1
                    loss += criterion(-outputs, fake_labels)  # Fake pairs should output 0
                    loss = loss / 2  # Average the two losses
                    
                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                    if i % 10 == 0:
                        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                            f"Loss: {loss.item():.4f}, "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                except Exception as e:
                    print(f"Error in training batch: {str(e)}")
                    continue
            
            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_epoch_loss)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_path = Path('model')
                save_path.mkdir(exist_ok=True)
                torch.save(
                    model.state_dict(),
                    save_path / 'pairwise_detector_best.pth'
                )
                print(f"New best model saved! Loss: {best_loss:.4f}")
        
        print("Training complete!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()