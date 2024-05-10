import ot
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from config import get_data
from utils.eval import evaluate, write_result
from tqdm import tqdm
from ot_gradient import auto_ot

from miss_methods.miss_dag_nonlinear import mean_imputation

def otm(X_init, lambda1, max_iter=100, h_tol=1e-8, rho_max=1e+16, eta=0.01):
    class Sup: 
        def __init__(self, X_init):
            mask = np.isnan(X_init)
            X_init[mask] = 0
            self.mask = mask.astype('float')
            self.X_init = X_init
            self.w = None # 2 * d * d
            self.imps = None # n,d

            

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

    def _wfunc(w):
        
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""

        imps = supp.imps.reshape(d,d)
        X = supp.X_init * (1 - supp.mask) + (supp.X_init @ imps)  * supp.mask
        X = X - np.mean(X, axis=0, keepdims=True) # for l2 only
        W = _adj(w)
        M = X @ W
        R = X - M
        loss = _loss(R)
        h, G_h = _h(W)      

        ot_loss, G = auto_ot(X, W, 'w')

        # Objective function
        obj = loss + alpha * h + lambda1 * w.sum() + 0.5 * rho * h * h + eta * ot_loss
        
        # Calculating gradient for W
        G_W = - 1.0 / X.shape[0] * X.T @ R
        G_smooth = G_W + (rho * h + alpha) * G_h + eta * G
    
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj , g_obj
    
    def _xfunc(imps):
        

        imps = imps.reshape(d, d)
        X = supp.X_init * (1 - supp.mask) + (supp.X_init @ imps) * supp.mask
        X = X - np.mean(X, axis=0, keepdims=True) # for l2 only
        W = _adj(supp.w)
        M = X @ W
        R = X - M

        ot_loss, G = auto_ot(X, W, 'x')
        
        obj = _loss(R) + eta * ot_loss
        
        # Calculating gradient
        I = np.eye(d,d)
        g_obj = 1.0 / X.shape[0] * (R @ (I - W.T)) * supp.mask
        g_obj = (supp.X_init.T @ g_obj) + eta * (supp.X_init.T @ G)
        g_obj = g_obj.reshape(-1)

        
        return obj , g_obj

    n, d = X_init.shape
    w_est = np.zeros(2 * d * d)
    
    rho, alpha, h = 1.0, 0.0, np.inf 
    imps = np.ones(d * d)
    
    wbnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    ibnds = [(None, None)] * imps.shape[0]

    supp.w = w_est 
    supp.imps = imps
        
    for i in range(max_iter):
        print(f'Iteration {i} ...')
        params_new, w_new, h_new = None, None, None
        while rho < rho_max:  
            sol = sopt.minimize(_xfunc, imps, method='L-BFGS-B', jac=True, bounds=ibnds)
            imps_new = supp.imps = sol.x  
            sol = sopt.minimize(_wfunc, w_est, method='L-BFGS-B', jac=True, bounds=wbnds)
            w_new = supp.w = sol.x 
                      
            print(imps_new.max().round(5), imps_new.min().round(5), w_new.sum().round(5))
            h_new, _ = _h(_adj(w_new))
            
            if h_new > 0.25 * h:
                rho *= 2
            else:
                break
        imps, w_est, h = imps_new, w_new, h_new
        params = params_new
        
        print(f'Current h={h}')
        
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            print(f'Stopping at h={h} and rho={rho}')
            break
    
    W_est = _adj(w_est)

    imps = imps.reshape(d,d)
    X_filled = supp.X_init * (1 - supp.mask) + (supp.X_init @ imps) * supp.mask
    return W_est, X_filled, supp.mask


if __name__ == '__main__':
    import sys
    from dag_methods import Notears

    
    config_id = int(sys.argv[1])
    graph_type = sys.argv[2]
    sem_type = 'linear'
    
    dataset, config = get_data(config_id, graph_type, sem_type)
    
    n,d = dataset.X.shape

    W_est, X_filled, mask = otm(dataset.X, lambda1=0.1, 
                                max_iter=10, h_tol=1e-8, rho_max=1e+16, eta=0.01)
 
    raw_result = evaluate(dataset.B_bin, W_est, threshold = 0.3)

    
    # =============== WRITE GRAPH ===============
    saved_path = f'output/otm_linear.txt'
    write_result(raw_result, config['code'], saved_path)


