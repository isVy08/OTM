import torch, os
import random
from utils.io import load_model


def free_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def model_loss(batch, model, criterion):
    x, f, h_val, reg = model(batch)
    loss = criterion.loss_fn(f, x)

    if 'mot' in criterion.methods:
        sample = random.sample(range(model.N), k = len(batch))
        xs = model.impute(sample)
        loss = loss + criterion.alpha * criterion.ot_dist(xs, x)
    
    if 'got' in criterion.methods:
        loss = loss + criterion.alpha * criterion.ot_dist(f, x)

    if criterion.beta is not None:
        loss = loss + criterion.beta * h_val
    
    if criterion.gamma is not None:
        loss = loss + criterion.gamma * reg
    return loss

def train_lbfgs(model, optimizer, loader, criterion):
 
    model.train()
   
    losses = 0
    for batch in loader:   

        def closure():
            optimizer.zero_grad()
            loss = model_loss(batch, model, criterion)
            loss.backward()
            return loss
        
        loss = model_loss(batch, model, criterion)
        optimizer.step(closure)
        losses += loss

    model.eval()
    with torch.no_grad():
        X_filled = model.impute(None)
            

    return X_filled, losses / len(loader)




def train_epoch(model, optimizer, loader, criterion):
    
    model.train()
   
    losses = 0
    for batch in loader:   

        loss = model_loss(batch, model, criterion)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss

    model.eval()
    with torch.no_grad():
        X_filled = model.impute(None)
            

    return X_filled, losses / len(loader)


