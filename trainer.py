import torch, os
import random
from utils.io import load_model


def free_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def train_epoch(model, optimizer, loader):
    
    model.train()
   
    losses = 0
    for batch in loader:   

        loss = model(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss

    model.eval()
    with torch.no_grad():
        X_filled = model.impute(None)
            

    return X_filled, losses / len(loader)


