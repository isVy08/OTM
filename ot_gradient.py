import ot 
import torch
import numpy as np

def manual_ot(X, W):
    M = X @ W
    C = 0.5 * ot.dist(X,M, metric='sqeuclidean')
    
    unif = np.ones((X.shape[0],))
    P = ot.emd(unif, unif, C) # solve for optimal transport plan
    loss = (C * P).mean() 
    G_W = 0
    I = np.eye(X.shape[1],X.shape[1])
    G_X = []

    for i in range(X.shape[0]): 
        G_Xai = 0
        # itself (Xai - XaiW)^2
        # by row (Xai - XajW)^2
        grad_row = 0
        # by column (Xaj - XaiW)^2
        grad_col = 0
        grad_self =  1.0 * ((X[i:i+1, ] - M[i:i+1, ]) @ (I - W.T)) * P[i,i]
        for j in range(n):
            G_W += -1.0 * (X[j:j+1, ].T @ (X[i:i+1, ] - M[j:j+1, ])) * P[i,j] 
            if i != j:
                grad_row +=  1.0 * ((X[i:i+1, ] - M[j:j+1, ]) @ I) * P[i,j]
                grad_col +=  1.0 * ((X[j:j+1, ] - M[i:i+1, ]) @ (-W.T)) * P[j,i]
        G_Xai = grad_self + grad_row + grad_col
        G_X.append(G_Xai)

    G_W = G_W / (X.shape[0] * X.shape[0])
    G_X = np.concatenate(G_X, axis = 0) / (X.shape[0] * X.shape[0])

    
    return loss, G_W, G_X

def auto_ot(Xa, Wa, wrt='x'):
    
    X = torch.from_numpy(Xa)
    X.requires_grad = True
    W = torch.from_numpy(Wa)
    W.requires_grad = True

    M = X @ W
    C = 0.5 * ot.dist(X,M, metric='sqeuclidean')

    # Optimal transport plan
    unif = torch.ones((X.shape[0],)) / X.shape[0]
    P = ot.emd(unif, unif, C)

    loss = (C * P).mean() 
    loss.backward()
    loss = loss.detach().numpy()
    if wrt == 'x':
        g = X.grad.detach().numpy()
        return loss, g
    else:
        g = W.grad.detach().numpy()
        return loss, g
    
