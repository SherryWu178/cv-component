import torch
import trainer
from utils import parse_arguments, count_classes
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
import cv2
import argparse

# class ResNet(nn.Module):
#     def __init__(self, num_classes, args) -> None:
#         super().__init__()

#         self.resnet = torchvision.models.resnet18(pretrained=True)
#         ##################################################################
#         # TODO: Define a FC layer here to process the features
#         ##################################################################
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

#         for param in self.resnet.parameters():
#             param.requires_grad = False

        
#         # Define your own classification layer for the specific task
#         self.fc = nn.Linear(512, num_classes) 
#         self.fc.requires_grad = True

#         self.softmax = nn.Softmax(dim=1)

#         ##################################################################
#         #                          END OF YOUR CODE                      #
#         ##################################################################
        

#     def forward(self, x):
#         ##################################################################
#         # TODO: Return unnormalized log-probabilities here
#         ##################################################################
#         # Flatten the output (if needed) before passing it to the classification layer
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)

#         # Pass the features through your classification layer
#         out = self.fc(x)
#         return self.softmax(out)
#         ##################################################################
#         #                          END OF YOUR CODE                      #
#         ##################################################################


import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Initialize ResNet without specifying weights and load pre-trained weights
        self.resnet = torchvision.models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze pre-trained parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Define your own classification layer for the specific task
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through the classification layer
        out = self.fc(x)
        return out


class ResNetWithEdge(nn.Module):
    def __init__(self, num_classes, args):
        super(ResNetWithEdge, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512 * 3, 256)  # Added fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(256, num_classes)  # Classification layer
        # self.softmax = nn.Softmax(dim=1)

        self.canny_low = args.canny_low
        self.canny_high = args.canny_high

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Canny edge detection
        edge_masks_1 = torch.zeros_like(x[:, 0:1, :, :])  # Initialize edge masks tensor
        edge_masks_2 = torch.zeros_like(x[:, 0:1, :, :])  # Initialize edge masks tensor
        for i in range(x.size(0)):
            # Convert tensor image to numpy array
            img_np = x[i].permute(1, 2, 0).cpu().numpy()
            # Convert to grayscale and ensure dtype is uint8
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.uint8)
            # Apply Canny edge detection
            edges_1 = cv2.Canny(gray_img, 100 , 200)
            edges_2 = cv2.Canny(gray_img, self.canny_low , self.canny_high)

            # Convert numpy array back to tensor and normalize
            edge_masks_1[i:i+1] = torch.from_numpy(edges_1).unsqueeze(0).unsqueeze(0) / 255.0
            edge_masks_2[i:i+1] = torch.from_numpy(edges_2).unsqueeze(0).unsqueeze(0) / 255.0

        # Concatenate original image with edge mask
        x_with_edge_1 = torch.cat((edge_masks_1.to(x.device), edge_masks_1.to(x.device), edge_masks_1.to(x.device)), dim=1)
        x_with_edge_2 = torch.cat((edge_masks_2.to(x.device), edge_masks_2.to(x.device), edge_masks_2.to(x.device)), dim=1)

        # Pass through ResNet separately for original and masked images
        features_original = self.resnet(x)
        features_original = features_original.view(features_original.size(0), -1)

        features_masked_1 = self.resnet(x_with_edge_1)
        features_masked_1 = features_masked_1.view(features_masked_1.size(0), -1)
        
        features_masked_2 = self.resnet(x_with_edge_2)
        features_masked_2 = features_masked_2.view(features_masked_2.size(0), -1)

        # Concatenate features
        features_combined = torch.cat((features_original, features_masked_1, features_masked_2), dim=1)

        # Apply first fully connected layer and ReLU activation function
        features_combined = self.fc1(features_combined)
        features_combined = self.relu(features_combined)

        # Classification
        # out = self.softmax(self.fc2(features_combined))
        out = self.softmax(self.fc2(features_combined))
        return out

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # parse the paramaters
    args = parse_arguments()

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    # model = ResNet(len(VOCDataset.classes)).to(args.device)
    num_classes = count_classes(args.data_dir)
    model = ResNet(num_classes).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
