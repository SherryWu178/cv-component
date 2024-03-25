from io import BytesIO
import numpy as np
from base64 import b64decode
import torch
import torchvision.transforms as transforms
import argparse
import os
import cv2
from voc_dataset import VOCDataset  
from IPython.display import display, Javascript
from google.colab.output import eval_js

def load_model(model_path):
    from train_q2 import ResNet

    num_classes = len(VOCDataset.classes)
    model = ResNet(num_classes)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model.eval()

    return model

#Define command line arguments
parser = argparse.ArgumentParser(description='Image classification using a trained model')
parser.add_argument('--model_path', type=str, help='Name of the model', required=True)
args = parser.parse_args()

#Load the model
model = load_model(args.model_path)

#Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model's expected input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5]),  # Normalize
])

def process_image(data):
    binary = b64decode(data.split(',')[1])
    image = Image.open(BytesIO(binary)).convert('RGB')
    return image

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video : true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight,
                                                true);

            // Wait for Capture to be clicked.
            await new Promise((resolve) = > capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
                }
                ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

#Inference loop
while True:
    # try:
    data = eval_js('takePhoto({})'.format(0.8))

    #Process image
    image = process_image(data)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # #Perform inference
    # with torch.no_grad():
    #     output = model(image_tensor)

    # #Get the predicted label
    # _, predicted = torch.max(output, 1)
    # print("Predicted label:", predicted.item())

    #Display the captured image
    display(image)

    # except Exception as e:
    #     print("An error occurred during inference:", e)
    #     break
