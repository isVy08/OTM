import torch
import torch.nn as nn

class Sample_Categorical(nn.Module):
    
  def __init__(self, tau):
    super(Sample_Categorical, self).__init__()
    self.tau = tau
  
  def forward(self, logits):
    # logits : [B, K, 1], K categories
    logits = logits.squeeze(-1)
    c = logits.size(-1)
    
    if self.training:
      sample = nn.functional.gumbel_softmax(logits, hard=True, tau=self.tau, dim=-1)
    else:
      choice = torch.distributions.categorical.Categorical(logits=logits)
      s = choice.sample()
      sample = torch.eye(c)[s]
      sample = sample.to(logits.device)
    return sample

class Sample_Bernoulli(nn.Module):
  def __init__(self, tau):
    super(Sample_Bernoulli, self).__init__()
    self.tau = tau

  def forward(self, probs):
    if self.training:
      
      unif_a = torch.rand(probs.shape).to(probs.device)
      gumbel_a = -torch.log(-torch.log(unif_a))
      no_logits = (probs * torch.exp(gumbel_a))/self.tau
      
      unif_b = torch.rand(probs.shape).to(probs.device)
      gumbel_b = -torch.log(-torch.log(unif_b))
      de_logits = no_logits + ((1.0 - probs) * torch.exp(gumbel_b))/self.tau
      
      sample = no_logits / de_logits

    else:
      probs = torch.clamp(probs, min=0.0, max=1.0)
      sample = torch.bernoulli(probs)
      sample = sample.to(probs.device)

    return sample


class Sample_Dirichlet(nn.Module):
    def __init__(self):
        super(Sample_Dirichlet, self).__init__()

    def forward(self, alpha):
        # [B, K], alpha > 0
        K = alpha.size(1)
        log_alpha = torch.log(alpha)
        mu = log_alpha - torch.mean(log_alpha)
        sum_inv = (1 / alpha).sum() 
        std = (1 / alpha) * (1 - 2/K) + 1/(K*K) * sum_inv
        std = torch.sqrt(std)
        eps = torch.randn_like(std, device = mu.device)
        logit = eps * std + mu
        return torch.softmax(logit, dim = -1)

