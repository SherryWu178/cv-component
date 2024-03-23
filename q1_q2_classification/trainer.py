from __future__ import print_function

import torch
import numpy as np
import utils
from voc_dataset import VOCDataset
import wandb

def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
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

def train(args, model, optimizer, scheduler=None, model_name='model'):
    wandb.init(project="kit cv", name=model_name, config=vars(args))
    train_loader = utils.get_data_loader(
        args, 'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    # test_loader = utils.get_data_loader(
    #     'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        args, 'voc', train=False, batch_size=len(test_dataset), split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            loss = custom_multi_label_loss(output, target, wgt)
            
            # Log the loss to wandb
            wandb.log({"loss": loss.item(), "epoch": epoch, "cnt": cnt})
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            
            # print progress bar
            print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))

            optimizer.step()
            cnt += 1
                    # Validation iteration
        if epoch % args.val_every == 0:
            all_predictions = []
            all_targets = []
            model.eval()
            ap, map = utils.eval_dataset_map(model, args.device, test_loader)
            print("map: ", map)
            wandb.log({"map": map, "ap": ap, "epoch": epoch, "cnt": cnt})
            writer.add_scalar("map", map, cnt)
            model.train()

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
