import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
from torchvision.models import densenet169, DenseNet169_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)


num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)


model.load_state_dict(torch.load("pneumonia_densenet169.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()  

    return "YES (Pneumonia Detected)" if prediction > 0.5 else "NO (Normal)"


def predict_multiple_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No valid images found in the folder.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        result = predict_image(image_path)
        print(f"Image: {image_file} → Prediction: {result}")


test_folder = r"C:\Users\HP\Desktop\add_models\Pneumonia\test"  
predict_multiple_images(test_folder)
