import os
import torch
import torch.nn as nn
from torchvision.utils import save_image


nz = 100  
ngf = 64  
nc = 3    
ngpu = 1  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs("data/generated_images", exist_ok=True)


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


netG = Generator(ngpu).to(device)


model_path = "model/dcgan_generator.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

netG.load_state_dict(torch.load(model_path))
netG.eval()


with torch.no_grad():
    
    noise = torch.randn(64, nz, 1, 1, device=device)
    fake_images = netG(noise)

    
    output_path = "data/generated_images/new_fake_images.png"
    save_image(fake_images, output_path, normalize=True, nrow=8)
    print(f"Generated images saved at {output_path}")
