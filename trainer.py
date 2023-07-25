import torch
import numpy as np
import ot


def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())


def train_epoch(X, model, optimizer, mask, imps, loader, 
                batchsize, device, lda):
    

    n, d = X.shape    
    total = 0
    loss_fn = torch.nn.MSELoss()

    for idx1 in loader:

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps
        
        idx2 = np.random.choice(n, batchsize, replace=False)

        X1 = X_filled[idx1].to(device)
        X2 = X_filled[idx2].to(device)
        F1 = model(X1)
        unif = torch.ones((batchsize,), device = device) / batchsize
        M = torch.zeros((batchsize, batchsize), device = device)
        for i in range(batchsize):
            for j in range(batchsize):
                M[i, j] = loss_fn(X1[i:i+1, ], X2[j:j+1, ]) + loss_fn(F1[i:i+1,], X2[j:j+1,])

    
        ws = ot.emd2(unif, unif, M)
        hW = torch.exp(torch.multiply(model.linear.weight, model.linear.weight))
        reg = torch.trace(hW) - d # + model.linear.weight.norm(dim=1,p=1).abs().sum()
        loss = ws + lda * reg + loss_fn(X1, F1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += loss 


    return X_filled, total / len(loader)


