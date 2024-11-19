from flask import Flask, request, jsonify, render_template
import torch
import traceback
import logging
from torchvision import transforms, models
from PIL import Image
import io
import torch.nn as nn

# Define Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing
image_size = 224
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Load the trained model
def load_model():
    global model
    try:
        model = DeepFakeDetector().to(device)
        model.load_state_dict(torch.load('deepfake_detector.pth', map_location=device))
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        traceback.print_exc()

# Preprocess image
def process_image(image_file):
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
        return image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global model
    if model is None:
        return jsonify({"error": "Model not initialized"}), 503

    try:
        test_image_file = request.files.get('test_image')

        if not test_image_file:
            return jsonify({"error": "Test image is required."}), 400

        test_image = process_image(test_image_file)

        if test_image is None:
            return jsonify({"error": "Failed to process image."}), 400

        # Perform inference
        with torch.no_grad():
            output = model(test_image)
            score = torch.sigmoid(output).item()
            is_fake = score < 0.5  # Adjust threshold as needed
            confidence = abs(score - 0.5) * 2  # Confidence range from 0 to 1

        response = {
            "is_fake": is_fake,
            "score": score,
            "confidence": confidence
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during detection: {e}")
        traceback.print_exc()
        return jsonify({"error": "An error occurred during detection."}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
