import ot
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from config import get_data
from utils.eval import evaluate, write_result
from tqdm import tqdm

def otm(X_init, lambda1, max_iter=100, h_tol=1e-8, rho_max=1e+16, beta = None):
    print('Optimizing for OT:', beta is not None)
    class Sup: 
        def __init__(self, X_init):
            mask = np.isnan(X_init)
            X_init[mask] = 0
            self.mask = mask.astype('float')
            self.X_init = X_init

    global supp
    supp = Sup(X_init)

    def _loss(R):
        """Evaluate value and gradient of loss."""

        loss = 0.5 / R.shape[0] * (R ** 2).sum()
        return loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W) 
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _ot(X, W):
        M = X @ W
        # C = np.zeros((n, n)) # cost_matrix
        # for i in range(n): 
        #     for j in range(n):
        #         C[i,j] = 0.5 * ((X[i:i+1, ] - M[j:j+1, ]) ** 2).sum() 
        C = 0.5 * ot.dist(X,M, metric='sqeuclidean')
        
        unif = np.ones((n,))
        P = ot.emd(unif, unif, C) # solve for optimal transport plan
        loss = (C * P).mean() 
        G_W = 0
        I = np.eye(d,d)
        G_X = []

        for i in tqdm(range(n)): 
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

        G_W = G_W / (n * n)
        G_X = np.concatenate(G_X, axis = 0) / (n*n)

        
        return loss, G_W, G_X


    def _func(params):
        
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        w = params[:d*d*2] 
        imps = params[d*d*2:].reshape(n,d)
        X = supp.X_init * (1 - supp.mask) + imps * supp.mask
        X = X - np.mean(X, axis=0, keepdims=True) # for l2 only
        W = _adj(w)
        M = X @ W
        R = X - M
        loss = _loss(R)
        h, G_h = _h(W)       
        # Objective function
        obj = loss + alpha * h + lambda1 * np.abs(w).sum() + 0.5 * rho * h * h
        
        # Calculating gradient
        G_W = - 1.0 / X.shape[0] * X.T @ R
        I = np.eye(d,d)
        G_imps = 1.0 / X.shape[0] * (R @ (I - W.T)) * supp.mask
        
        G_smooth = G_W + (rho * h + alpha) * G_h
        # If using OT
        if beta is not None: 
            ot_loss, G_W_ot, G_imps_ot = _ot(X, W)
            obj += beta * ot_loss
            G_imps += (G_imps_ot * supp.mask)
            G_smooth += beta * G_W_ot

        G_imps = G_imps.reshape(-1,) 
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        g_obj = np.concatenate((g_obj, G_imps.reshape(-1, )), axis = None)
        return obj , g_obj
    

    n, d = X_init.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    imps = np.random.randn(n,d).reshape(-1, )
    params = np.concatenate((w_est, imps), axis = 0)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)] + [(None, None)] * imps.shape[0]
        
    for i in range(max_iter):
        print(f'Iteration {i} ...')
        params_new, w_new, h_new = None, None, None
        while rho < rho_max:  
            sol = sopt.minimize(_func, params, method='L-BFGS-B', jac=True, bounds=bnds, options={'maxiter': 50})
            params_new = sol.x 
            w_new = params_new[:2*d*d]
            imps_new = params_new[2*d*d:]
            print(imps_new.max().round(5),
                imps_new.min().round(5),
                w_new.sum().round(5))
            h_new, _ = _h(_adj(w_new))
            
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        imps, w_est, h = imps_new, w_new, h_new
        params = params_new
        
        print(f'Current h={h}')
        
        # alpha += rho * h
        # if rho >= rho_max:
        #     print('Resetting rho')
        #     rho = 1.0
        if h <= h_tol or rho >= rho_max:
            print(f'Stopping at h={h} and rho={rho}')
            break
    
    W_est = _adj(w_est)
    imps = imps.reshape(n,d)
    X_filled = supp.X_init * (1 - supp.mask) + imps * supp.mask
    # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, X_filled, supp.mask


if __name__ == '__main__':
    import sys
    
    config_id = int(sys.argv[1])
    graph_type = sys.argv[2] # ER, SF
    dataset, config = get_data(config_id, graph_type, 'linear')
    
    lambda1 = 0.1
    n,d = dataset.X.shape
    W_est, X_filled, mask = otm(dataset.X, lambda1, max_iter=30, h_tol=1e-8, rho_max=1e+10, beta = 0.1)
    
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)
    
    # =============== WRITE GRAPH ===============
    saved_path = 'output/otm_linear_wot.txt'
    write_result(raw_result, config['code'], saved_path)

    print(X_filled)

    from utils.missing import MAE, RMSE
    mae = MAE(X_filled, dataset.X_true, mask)
    rmse = RMSE(X_filled, dataset.X_true, mask)

    # =============== WRITE IMPUTATION ===============
    file = open('output/otm_imputation_wot.txt', 'a+')
    file.write(f'{config["code"]}\n')
    file.write(f'MAE: {mae}, RMSE: {rmse}\n')
    file.write('======================\n')
    file.close()

    

    
    # loss_type = 'l2'
    # max_iter=100
    # h_tol=1e-8
    # rho_max=1e+16
    # w_threshold=0.3


