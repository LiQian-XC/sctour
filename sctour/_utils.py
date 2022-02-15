import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import issparse


##calculate KL divergence
def normal_kl(mu1, lv1, mu2, lv2):
    """
    Calculate KL divergence
    This function is from torchdiffeq: https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py
    """
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1/2.
    lstd2 = lv2/2.

    kl = lstd2 - lstd1 + (v1 + (mu1-mu2)**2.)/(2.*v2) - 0.5
    return kl


## get step size
def get_step_size(step_size, t1, t2, t_size):
    if step_size is None:
        options = {}
    else:
        step_size = (t2 - t1)/t_size/step_size
        options = dict(step_size = step_size)
    return options


##calculate log zinb probability
def log_zinb(x, mu, theta, pi, eps=1e-8):
    """
    Calculate log probability under zero-inflated negative binomial distribution
    This function is from scvi-tools: https://github.com/YosefLab/scvi-tools/blob/6dae6482efa2d235182bf4ad10dbd9483b7d57cd/scvi/distributions/_negative_binomial.py
    """
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


##calculate log nb probability
def log_nb(x, mu, theta, eps=1e-8):
    """
    Calculate log probability under negative binomial distribution
    This function is from scvi-tools: https://github.com/YosefLab/scvi-tools/blob/6dae6482efa2d235182bf4ad10dbd9483b7d57cd/scvi/distributions/_negative_binomial.py
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res


##L2 norm
def l2_norm(x, axis=-1):
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis = axis))
