import argparse
import torch
import torchvision.transforms as transforms
import os
import cv2
from voc_dataset import VOCDataset  # Assuming you have defined your classes in this file

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5]),  # Normalize
])

# Define command line arguments
parser = argparse.ArgumentParser(description='Image classification using a trained model')
parser.add_argument('--model_path', type=str, help='Name of the model', required=True)
args = parser.parse_args()

# Load the model
model = load_model(args.model_path)

# Open the default camera (usually the laptop camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Inference loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        try:
            # Convert the frame to PIL image format
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Apply transformations
            image = transform(pil_image)
            image = image.unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                output = model(image)

            # Get the predicted label
            _, predicted = torch.max(output, 1)
            print("Predicted label:", predicted.item())

        except Exception as e:
            print("An error occurred during inference:", e)

    # Display the captured frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
