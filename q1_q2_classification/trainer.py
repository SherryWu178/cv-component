from __future__ import print_function
from sklearn.metrics.cluster import entropy

import torch
import torch.nn as nn
import numpy as np
import utils
from voc_dataset import VOCDataset
import wandb
import os

def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, save_folder, model):
    os.makedirs(os.path.join("checkpoints", save_folder), exist_ok=True)
    filename = os.path.join("checkpoints", save_folder, f"checkpoint-{epoch+1}.pth")
    print("Saving model at:", filename)
    
    # Save the model state dictionary
    torch.save(model.state_dict(), filename)

def custom_multi_label_loss(output, target, wgt):
    """
    Custom loss function for multi-label classification with sample weights.
    
    Args:
    - output: Model predictions (logits)
    - target: Ground truth labels (binary, 0 or 1)
    - wgt: Weights for each sample (0 for easy, 1 for difficult)
    
    Returns:
    - loss: Computed loss, a single floating point number
    """
    # Apply sigmoid activation to logits to get predicted probabilities
    predicted_probs = torch.sigmoid(output)
    
    # Avoid potential numerical instability by clipping probabilities
    predicted_probs = torch.clamp(predicted_probs, min=1e-7, max=1 - 1e-7)
    
    # Compute the binary cross-entropy loss element-wise
    loss = -wgt * (target * torch.log(predicted_probs) + (1 - target) * torch.log(1 - predicted_probs))
    
    # Compute the average loss across all samples
    loss = torch.mean(loss)
    return loss

def train(args, model, optimizer, scheduler=None, model_name='model', num_classes=10):
    print("trainer num_classes", num_classes)
    wandb.init(project="kit cv", name=args.run_name, config=vars(args))
    train_loader = utils.get_data_loader(
        args, 'voc', train=True, batch_size=args.batch_size, split='train', inp_size=args.inp_size)

    val_loader = utils.get_data_loader(
        args, 'voc', train=False, batch_size=args.test_batch_size, split='val', inp_size=args.inp_size)
    
    test_loader = utils.get_data_loader(
        args, 'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            # Example of target with class indices
            criterion = nn.CrossEntropyLoss()

            predictions = torch.argmax(output, 1) 
            target_label = torch.argmax(target, 1)
            
            # loss = entropy(output, target_label)
            loss = criterion(output, target_label)
            accuracy = (predictions == target_label).sum().item() / len(target_label)

            lr = optimizer.param_groups[0]['lr']  # Assuming there's only one parameter group
            wandb.log({"Loss/train": loss, "learning_rate": lr, "cnt": cnt, "arruracy": accuracy})
            
            loss.backward()
            
            # print progress bar
            # if cnt % args.batch_size == 0:
            print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item(), accuracy))

            optimizer.step()
            cnt += 1
        
        # Validation iteration
        if epoch % args.val_every == 0:
            all_predictions = []
            all_targets = []
            model.eval()
            ap, map = utils.eval_dataset_map(model, args.device, val_loader, epoch, num_classes)
            print("map: ", map)
            wandb.log({"map": map, "ap": ap, "epoch": epoch, "cnt": cnt})
            model.train()

        if scheduler is not None:
            scheduler.step()

        # save model
        if save_this_epoch(args, epoch):
            # model_name = "latest-model"
            save_folder = args.run_name
            save_model(epoch, save_folder, model)

    # Final Validation
    ap, map = utils.eval_dataset_map(model, args.device, test_loader, epoch, num_classes)
    wandb.finish()
    return ap, map
