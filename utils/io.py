import pickle
import torch
import numpy as np


def load_txt(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data


def load_pickle(datadir):
    file = open(datadir, 'rb')
    data = pickle.load(file)
    return data


def write_pickle(data, savedir):
    file = open(savedir, 'wb')
    pickle.dump(data, file)
    file.close()


def load_model(model, optimizer, scheduler, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['prev_loss'], checkpoint['imputes'] 
