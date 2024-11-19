import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import streamlit as st

# Model parameters
nz = 100  # Latent vector size (input noise size for the original DCGAN)
ngf = 64  # Generator feature map size
nc = 3    # Number of channels (RGB)
ngpu = 1  # Number of GPUs

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_path = "model/dcgan_generator.pth"
output_dir = "data/generated_images"
os.makedirs(output_dir, exist_ok=True)

# Define the Generator architecture
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Initialize and load the generator model
netG = Generator(ngpu).to(device)
try:
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("Deepfake Image Generator")
st.write("Upload a real image to generate a deepfake image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    real_image = Image.open(uploaded_file).convert("RGB")
    st.image(real_image, caption="Uploaded Real Image", use_column_width=True)

    # Apply transformation
    real_image_tensor = transform(real_image).unsqueeze(0).to(device)

    # Generate fake image
    with torch.no_grad():
        # Generate noise vector
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = netG(noise)

        # Save the generated fake image
        output_path = os.path.join(output_dir, "generated_images.png")
        save_image(fake_image, output_path, normalize=True)

        # Display the generated fake image
        st.image(output_path, caption="Generated Deepfake Image", use_column_width=True)
        st.success(f"Generated image saved at {output_path}")
