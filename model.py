import torch, ot
import torch.nn as nn
import numpy as np
from utils.arch import MLP
from utils.missing import nanmean
from dagma import DagmaMLP, DagmaLinear
from tqdm.auto import tqdm
import typing, copy, random

class SuperImputer(nn.Module): 
    def __init__(self, data, mask, hidden_dims, initialized = None):
        super(SuperImputer, self).__init__()
        print('Using Super imputation ...')
        self.D = hidden_dims[0]        
        self.mu = MLP(hidden_dims, nn.ReLU())
        self.var = MLP(hidden_dims, nn.ReLU())

        self.data = data 
        self.mask = mask
        self.initialized = initialized

        if initialized == 'learnable':
            imps = (torch.randn(data.shape, device = mask.device) + nanmean(data, 0))[mask.bool()]
            self.imps = nn.Parameter(imps)
    
    def forward(self): 
        x = self.data.clone()  
        if self.initialized is None: # pre-imputed with zeros
            x[self.mask.bool()] = 0.0

        elif self.initialized == 'learnable':
            x[self.mask.bool()] = self.imps
        
        logvar = self.var(x)
        imps = self.mu(x) + torch.exp(0.5 * logvar) * torch.randn_like(x)
        x = imps * self.mask + x
        return x 
        
class SimpleImputer(nn.Module): 
    def __init__(self, data, mask):
        super(SimpleImputer, self).__init__()
        print('Using Simple imputation ...')
        imps = (torch.randn(data.shape, device = mask.device).float() + nanmean(data, 0))[mask.bool()]
        
        self.imps = nn.Parameter(imps)
        
        self.data = data 
        self.mask = mask
    
    def forward(self): 
        x = self.data.clone()  
        x[self.mask.bool()] = self.imps        
        return x

class MissModel(nn.Module):
    def __init__(self, data, mask, hidden_dims, device, sem_type, initialized = None):
        super(MissModel, self).__init__()

        self.d = hidden_dims[0]
        self.sem_type = sem_type
        
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)
        self.imputer = SuperImputer(data, mask, [self.d, self.d], initialized)
        
        
    def to_adj(self):
        return self.scm.fc1_to_adj()

    def forward(self):
        '''
        x : torch.Tensor shape (N,D)
        '''
        x = self.imputer()      
        # reconstruction from the imputations
        xhat = self.scm(x) 
        return x, xhat
    
class DagmaNonlinear:
    """
    Class that implements the DAGMA algorithm
    """
    
    def __init__(self, model: nn.Module, verbose: bool = False, dtype: torch.dtype = torch.double):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.double``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        self.data = model.imputer.data
        self.mask = model.imputer.mask
        self.model = model
        self.dtype = dtype
    
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def exact_ot_cost(self, X, sample_size = 10):
        '''
        Minimize cross-feature OT 
        X : N x D
        '''
        assert sample_size * 2 < self.model.d
        s1 = random.sample(range(self.model.d), k = sample_size)
        remain = [i for i in range(self.model.d) if i not in s1]
        s2 = random.sample(remain, k = sample_size)

        # Extract weighted adjacency matrix
        fc1_weight = self.model.scm.fc1.weight
        fc1_weight = fc1_weight.view(self.model.d, -1, self.model.d)
        W = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        cost_fn = torch.nn.functional.mse_loss

        dim  = sample_size
        unif = torch.ones((dim,), device = X.device) / dim
        
        C = torch.zeros((dim, dim), device = X.device) # must be nonnegative
        for i in range(dim): 
            for j in range(dim):
                a, b = s1[i], s2[j]
                weight = 1 - torch.sigmoid(W[a,b])
                ml = weight * cost_fn(X[:, a], X[:, b])
                C[i,j] = ml 
        
        loss = ot.emd2(unif, unif, C)
        return loss
    
    def rbf_kernel(self, X, Y):
        batch_size, h_dim = X.shape
    
        norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_x = torch.mm(X, X.t())  # batch_size x batch_size
        dists_x = norms_x + norms_x.t() - 2 * prods_x
        
        norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
        dists_y = norms_y + norms_y.t() - 2 * prods_y
        
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
        
        stats = 0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = 2 * h_dim * 1.0 / scale
            res1 = torch.exp(-C * dists_x) + torch.exp(-C * dists_y)

            res1 = (1 - torch.eye(batch_size).to(X.device)) * res1
            
            res1 = res1.sum() / (batch_size - 1)
            res2 = torch.exp(-C * dists_c)
            res2 = res2.sum() * 2. / batch_size
            stats += res1 - res2

        return stats / batch_size

    def minimize(self, 
                 max_iter: float, 
                 lr: float, 
                 lambda1: float, 
                 lambda2: float, 
                 mu: float, 
                 s: float,
                 lr_decay: float = False, 
                 tol: float = 1e-6, 
                 pbar: typing.Optional[tqdm] = None,
        ) -> bool:
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(.99,.999), weight_decay=mu*lambda2)

        if lr_decay is True:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            
            h_val = self.model.scm.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            
            X, Xhat = self.model()

            if self.model.sem_type == 'linear':
                score = torch.nn.functional.mse_loss(Xhat, X)
            else:
                score = self.log_mse_loss(Xhat, X)
            
            l1_reg = lambda1 * self.model.scm.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val 

            s1 = random.sample(range(X.shape[0]), k = 50)
            remain = [i for i in range(X.shape[0]) if i not in s1]
            s2 = random.sample(remain, k = 50)

            coefs = {'mlp': 0.01, 'gp': 0.01}
            obj = obj + coefs[self.model.sem_type] * self.rbf_kernel(X[s1, :], X[s2, :])
   
            optimizer.zero_grad()
            # obj.backward()
            obj.backward(retain_graph=True)
            optimizer.step()
            
            if lr_decay and (i+1) % 1000 == 0: #every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter-1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter-i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True

    def fit(self,
            lambda1: float = .02, 
            lambda2: float = .005,
            T: int = 4, 
            mu_init: float = .1, 
            mu_factor: float = .1, 
            s: float = 1.0,
            warm_iter: int = 5e4, 
            max_iter: int = 8e4, 
            lr: float = .0002, 
            w_threshold: float = 0.3, 
            checkpoint: int = 1000,
        ) -> np.ndarray:        
        
        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.") 
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i+1} -- mu: {mu}', 30*'-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur, 
                                        lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5 
                        lr_decay = True
                        if lr < 1e-10:
                            break # lr is too small
                        s_cur = 1
                mu *= mu_factor
        W_est = self.model.to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
