import torch
import utils
from torch.utils.data import DataLoader
from utils import ARGS
from train_q2 import ResNet

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.color import rgb2lab, deltaE_cie76
from collections import defaultdict

PATH = "/content/hw1/checkpoint-4(freezed weight)/checkpoint-model-epoch50.pth"
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
                   
def load_model(filename):
    return torch.load(filename)

def extract_features(model, device, test_loader):
    with torch.no_grad():
        test_features = []
        gt_classes = []
        for data, target, wgt in test_loader:
            data = data.to(device)
            output = model(data.cpu())
            test_features.append(output.cpu().numpy().ravel())
            gt_classes.append(target.cpu().numpy())

    test_features = np.array(test_features)
    return test_features, gt_classes


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def draw_plot(test_features, gt_classes):
    # Ensure that the features are standardized (mean=0, std=1)
    print(len(test_features))
    print(test_features[0].shape)
    # scaler = StandardScaler()
    # test_features = scaler.fit_transform(test_features)

    # Perform t-SNE on the features
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(test_features)
 
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    print("tsne_results")
    print(tsne_results)

    # Prepare colors for class labels
    class_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    class_colors_dic = {}
    for i in range(20):
        class_colors_dic[CLASS_NAMES[i]] = class_colors[i]


    # Create a scatter plot with color-coded points
    plt.figure(figsize=(10, 8))
    mean_colors = np.zeros((len(gt_classes), 4))  # Initialize a 2D array for colors (RGBA format)

    for i, class_label in enumerate(gt_classes):
        mask = (np.ones(20) == class_label.ravel())
        mean_color = np.mean(class_colors[mask], axis=0)
        mean_colors[i] = mean_color  # Assign the mean color to the 2D array
   
    print(len(mean_colors))
    print(mean_colors[0])
    plt.scatter(tx, ty, c=mean_colors)

    # Create a legend with class labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {label}', 
                        markersize=10, markerfacecolor=color) for label, color in class_colors_dic.items()]
    print(handles)
    plt.legend(handles=handles, title='Object Classes')

    # Add labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Projection of ImageNet Features Color-Coded by GT Class')

    
    file_path = "t-SNE plot.png"  
    # Save the plot to the specified file
    plt.savefig(file_path, dpi=300)

    # Close the plot (optional)
    plt.close()

    # Print a message indicating that the plot has been saved
    print(f"Plot saved as {file_path}")

def visualize(args, model, test_loader):
    # Determine the total number of samples in the test dataset
    total_samples = len(test_loader.dataset)

    # Set a random seed for reproducibility (optional)
    torch.manual_seed(27)

    # Randomly select 1000 indices from the test dataset
    selected_indices = torch.randperm(total_samples)[:1000]

    # Create a Subset of the test dataset containing the selected samples
    from torch.utils.data import Subset
    subset = Subset(test_loader.dataset, selected_indices)

    # Create a DataLoader for the selected samples
    selected_test_loader = DataLoader(subset, batch_size=test_loader.batch_size, shuffle=True)
    
    model = ResNet(20)
    model.load_state_dict(torch.load(PATH))
    finetuned_model = model.resnet
    test_features, gt_classes = extract_features(finetuned_model, args.device, selected_test_loader)
    draw_plot(test_features, gt_classes)

if __name__ == "__main__":
    args = ARGS(
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=70,
        lr=0.001,
        batch_size=256,
        step_size=10,
        gamma=0.1,
        save_at_end=True,
        save_freq =10,
        test_batch_size = 1
    )
    model = ResNet(20)
    model = load_model(PATH)
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    visualize(args, model, test_loader)
    