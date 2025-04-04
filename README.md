# DeepFake Detection 🕵️‍♂️🧠

A robust deep learning-based system for detecting AI-generated faces using pairwise learning and advanced CNN architectures. This project leverages real vs. GAN-generated image comparisons to effectively distinguish DeepFakes from authentic images.

## 🚀 Features

- 🔍 **Pairwise Learning**: Compares real and fake image pairs using a contrastive loss
- 🧬 **Custom CNN Backbone**: Utilizes DenseNet and CFFN for feature extraction
- 🎭 **Multi-GAN Dataset**: Trained on fakes generated from multiple GANs like DCGAN, CycleGAN, StyleGAN, etc.
- 🧠 **PyTorch-based**: Built with modular, scalable PyTorch code
- 📊 **Evaluation Tools**: Includes accuracy, AUC, precision-recall, and confusion matrix

## 📁 Project Structure

```
DeepFake-Detection/
│
├── data/                  # Real and Fake image directories
├── models/                # CNN, DenseNet, and CFFN models
├── utils/                 # Utility functions and dataset loaders
├── train.py               # Training script
├── test.py                # Evaluation script
├── config.py              # Configuration and hyperparameters
└── README.md              # You're here!
```

## 📦 Installation

```bash
git clone https://github.com/Taskmaster-1/DeepFake-Detection.git
cd DeepFake-Detection
pip install -r requirements.txt
```

Make sure you have a CUDA-enabled GPU and PyTorch installed.

## 🧠 Model Architecture

- **Input**: Pair of images (Real, Fake)
- **Backbone**: CFFN or DenseNet-based encoder
- **Head**: Fully connected layers + Sigmoid
- **Loss**: Contrastive loss / Binary Cross Entropy

## 🧪 Sample Fake Sources

- StyleGAN
- CycleGAN
- DCGAN
- ProGAN
- Celeb-DF

## 🙌 Acknowledgements

- FaceForensics++
- Kaggle Deepfake Datasets

## 📄 License

This project is licensed under the MIT License.
