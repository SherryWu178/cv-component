import os
import torch
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import wandb
import random
from voc_dataset import VOCDataset

def parse_arguments():
    default_device = torch.device("cuda")
    default_batch_size = 8
    default_test_batch_size = 1000
    default_epochs = 50
    default_lr = 0.00005
    default_gamma = 0.1
    default_step_size = 30
    default_log_every = 100
    default_val_every = 1
    default_save_at_end = True
    default_save_freq = 5
    default_use_cuda = True
    default_inp_size = 224
    default_canny_low = 100
    default_canny_high = 200
    default_data_dir = "data/new_set_up_frames/"
    default_run_name = "new_run"

    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--device', type=int, default=default_device, help='Device')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=default_test_batch_size, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=default_epochs, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=default_lr, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=default_gamma, help='Gamma value')
    parser.add_argument('--step_size', type=int, default=default_step_size, help='Step size for learning rate scheduler')
    parser.add_argument('--log_every', type=int, default=default_log_every, help='Frequency to log training status')
    parser.add_argument('--val_every', type=int, default=default_val_every, help='Frequency to evaluate model')
    parser.add_argument('--save_at_end', type=bool, default=default_save_at_end, help='Whether to save model at the end of training')
    parser.add_argument('--save_freq', type=int, default=default_save_freq, help='Frequency to save model during training')
    parser.add_argument('--use_cuda', type=bool, default=default_use_cuda, help='Whether to use CUDA for training')
    parser.add_argument('--inp_size', type=int, default=default_inp_size, help='Input size')
    parser.add_argument('--canny_low', type=int, default=default_canny_low, help='Canny low threshold')
    parser.add_argument('--canny_high', type=int, default=default_canny_high, help='Canny high threshold')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Directory where the dataset is located')
    parser.add_argument('--run_name', type=str, default=default_run_name, help='Given WandB a run name')

    args = parser.parse_args()
    print(args)
    return args

def get_data_loader(args, name='voc', train=True, batch_size=64, split="", inp_size=224):
    if name == 'voc':
        from voc_dataset import VOCDataset
        dataset = VOCDataset(split, inp_size, data_dir=args.data_dir)
    else:
        raise NotImplementedError

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    return loader


def compute_ap(gt, pred, valid, average=None):
    # ! Do not modify the code in this function
    
    """
    Compute the multi-label classification accuracy.
    Args:
        gt (np.ndarray): Shape Nx20, 0 or 1, 1 if the object i is present in that
            image.
        pred (np.ndarray): Shape Nx20, probability of that object in the image
            (output probablitiy).
        valid (np.ndarray): Shape Nx20, 0 if you want to ignore that class for that
            image. Some objects are labeled as ambiguous.
    Returns:
        AP (list): average precision for all classes
    """
    nclasses = gt.shape[1]
    print(nclasses)
    AP = []
    true_labels = []
    pred_labels = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        # As per PhilK. code:
        # https://github.com/philkr/voc-classification/blob/master/src/train_cls.py
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, device, test_loader, epoch, num_classes):
    # ! Do not modify the code in this function

    """
    Evaluate the model with the given dataset
    Args:
         model (keras.Model): model to be evaluated
         dataset (tf.data.Dataset): evaluation dataset
    Returns:
         AP (list): Average Precision for all classes
         MAP (float): mean average precision
    """

    true_labels = []
    pred_labels = []
    wrongly_predicted_samples = []
    
    with torch.no_grad():
        AP_list = [0] * num_classes
  
        
        for data, target, wgt in test_loader:
            data = data.to(device)
            output = model(data)
            
            pred = F.softmax(output, dim=-1).cpu().numpy()
            gt = target.cpu().numpy()
            valid = wgt.cpu().numpy()

            gt_label = np.argmax(gt, axis=1)
            pred_label = np.argmax(pred, axis=1)
            
            true_labels.extend(gt_label)
            pred_labels.extend(pred_label)
            
            AP = compute_ap(gt, pred, valid)

            print(len(AP))
            print(len(AP_list))

            for i in range(len(AP)):
                AP_list[i] += AP[i]
            
            for i in range(len(pred_label)):
                if pred_label[i] != gt_label[i]:
                    wrongly_predicted_samples.append((data[i].permute((1, 2, 0)), gt_label[i], pred_label[i]))


    AP_list = [ ap / len(test_loader) for ap in AP_list]
    confusion_matrix_result = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
    print("confusion_matrix_result", confusion_matrix_result)
    print(true_labels)
    print(pred_labels)
    print(torch.sum(torch.eq(torch.tensor(true_labels), torch.tensor(pred_labels))).item())
    accuracy = torch.sum(torch.eq(torch.tensor(true_labels), torch.tensor(pred_labels))).item() / len(pred_labels)
    AP = np.array(AP_list)
    print("AP", AP, "test accuracy", accuracy)
    mAP = np.mean(AP)

    wandb.log({"my_conf_mat_id" : wandb.plot.confusion_matrix( 
        preds=pred_labels, y_true=true_labels,
        class_names=VOCDataset.classes)})

    wandb.log({ "mAP": mAP, "epoch": epoch, "test arruracy": accuracy})
    
    if wrongly_predicted_samples:
        # Randomly sample 10 examples
        sampled_samples = random.sample(wrongly_predicted_samples, min(10, len(wrongly_predicted_samples)))
        wrongly_predicted_images = []

        for data, true_label, pred_label in sampled_samples:
            wrongly_predicted_images.append(wandb.Image(data.cpu().numpy(), caption=f"True Label: {true_label}, Predicted Label: {pred_label}"))

        wandb.log({"Wrongly_Predicted_Images": wrongly_predicted_images})
    else:
        print("No wrongly predicted samples to log.")

    return AP, mAP

def count_classes(root_folder):
    train_folder = os.path.join(root_folder, "train")
    # List all subdirectories in the train folder, excluding hidden folders
    subdirectories = [folder for folder in os.listdir(train_folder) if not folder.startswith('.')]
    num_classes = len(subdirectories)
    return num_classes

def get_class_names(root_folder):
    train_folder = os.path.join(root_folder, "train")
    class_names = os.listdir(train_folder)
    return sorted(class_names)


