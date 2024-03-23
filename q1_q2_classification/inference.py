import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from voc_dataset import VOCDataset

def load_model(model_path):
    # Assuming YourModelClass is defined in the file containing the model
    from train_q2 import ResNet

    # Instantiate your model class
    num_classes = len(VOCDataset.classes)
    model = ResNet(num_classes)

    # Load the trained model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # Set the model to evaluation mode
    model.eval()

    return model

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model's expected input size
    transforms.ToTensor(),  # Convert image to tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5]),  # Normalize
])

# Define command line arguments
parser = argparse.ArgumentParser(description='Image classification using a trained model')
parser.add_argument('--model_path', type=str, help='Name of the model', required=True)
args = parser.parse_args()

# Load the model
model = load_model(args.model_path)

# Inference loop
while True:
    # Ask for image path
    image_path = input("Enter the path to the image (or 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break

    # Check if the file exists
    if not os.path.exists(image_path):
        print("File not found.")
        continue

    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image)

        # Get the predicted label
        _, predicted = torch.max(output, 1)
        print("Predicted label:", predicted.item())

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print("An error occurred during inference:", e)
