import ot 
import torch
import numpy as np

def _ot(X, W):
    M = X @ W
    C = 0.5 * ot.dist(X,M, metric='sqeuclidean')
    
    unif = np.ones((X.shape[0],))
    P = ot.emd(unif, unif, C) # solve for optimal transport plan
    loss = (C * P).mean() 
    G_W = 0
    I = np.eye(X.shape[1],X.shape[1])
    G_X = []

    for i in range(X.shape[0]): 
        G_X_i = 0
        # itself (X_i - X_iW)^2
        # by row (X_i - X_jW)^2
        grad_row = 0
        # by column (X_j - X_iW)^2
        grad_col = 0
        grad_self =  1.0 * ((X[i:i+1, ] - M[i:i+1, ]) @ (I - W.T)) * P[i,i]
        for j in range(n):
            G_W += -1.0 * (X[j:j+1, ].T @ (X[i:i+1, ] - M[j:j+1, ])) * P[i,j] 
            if i != j:
                grad_row +=  1.0 * ((X[i:i+1, ] - M[j:j+1, ]) @ I) * P[i,j]
                grad_col +=  1.0 * ((X[j:j+1, ] - M[i:i+1, ]) @ (-W.T)) * P[j,i]
        G_X_i = grad_self + grad_row + grad_col
        G_X.append(G_X_i)

    G_W = G_W / (X.shape[0] * X.shape[0])
    G_X = np.concatenate(G_X, axis = 0) / (X.shape[0] * X.shape[0])

    
    return loss, G_W, G_X

def _auto_ot(X_, W_):
    
    X = torch.from_numpy(X_)
    X.requires_grad = True
    W = torch.from_numpy(W_)
    W.requires_grad = True

    M = X @ W
    C = 0.5 * ot.dist(X,M, metric='sqeuclidean')

    # Optimal transport plan
    unif = torch.ones((X.shape[0],)) / X.shape[0]
    P = ot.emd(unif, unif, C)

    loss = (C * P).mean() 
    loss.backward()
    G_W = W.grad.detach().numpy()
    G_X = X.grad.detach().numpy()
    return loss.detach().numpy(), G_W, G_X
