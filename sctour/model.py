import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint
from typing import Optional
from typing_extensions import Literal

from .module import LatentODEfunc, Encoder, Decoder
from ._utils import get_step_size, normal_kl, log_zinb, log_nb


class TNODE(nn.Module):
    """
    Class to automatically infer cellular dynamics using VAE and neural ODE.

    Parameters
    ----------
    n_int
        The dimensionality of the input.
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_ode_hidden
        The dimensionality of the hidden layer for the latent ODE function.
        (Default: 25)
    n_vae_hidden
        The dimensionality of the hidden layer for the VAE.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer.
        (Default: `False`)
    ode_method
        Solver for integration.
        (Default: `'euler'`)
    step_size
        The step size during integration.
    alpha_recon_lec
        Scaling factor for reconstruction loss from encoder-derived latent space.
        (Default: 0.5)
    alpha_recon_lode
        Scaling factor for reconstruction loss from ODE-solver latent space.
        (Default: 0.5)
    alpha_kl
        Scaling factor for KL divergence.
        (Default: 1.0)
    loss_mode
        The mode for calculating the reconstruction error.
        (Default: `'nb'`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 5,
        n_ode_hidden: int = 25,
        n_vae_hidden: int = 128,
        batch_norm: bool = False,
        ode_method: str = 'euler',
        step_size: Optional[int] = None,
        alpha_recon_lec: float = 0.5,
        alpha_recon_lode: float = 0.5,
        alpha_kl: float = 1.,
        loss_mode: Literal['mse', 'nb', 'zinb'] = 'mse',
    ):
        super().__init__()
        self.n_int = n_int
        self.n_latent = n_latent
        self.n_ode_hidden = n_ode_hidden
        self.n_vae_hidden = n_vae_hidden
        self.batch_norm = batch_norm
        self.ode_method = ode_method
        self.step_size = step_size
        self.alpha_recon_lec = alpha_recon_lec
        self.alpha_recon_lode = alpha_recon_lode
        self.alpha_kl = alpha_kl
        self.loss_mode = loss_mode

        self.lode_func = LatentODEfunc(n_latent, n_ode_hidden)
        self.encoder = Encoder(n_int, n_latent, n_vae_hidden, batch_norm)
        self.decoder = Decoder(n_int, n_latent, n_vae_hidden, batch_norm, loss_mode)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        Given the transcriptomes of cells, this function predicts the time and latent space of the cells, as well as reconstructs the transcriptomes.

        Parameters
        ----------
        x
            The input data.
        y
            The library size.

        Returns
        ----------
        5-tuple of :class:`torch.Tensor`
            Tensors for loss, including:
            1) total loss,
            2) reconstruction loss from encoder-derived latent space,
            3) reconstruction loss from ODE-solver latent space,
            4) KL divergence,
            5) divergence between encoder-derived latent space and ODE-solver latent space
        """

        ## get the time and latent space through Encoder
        T, qz_mean, qz_logvar = self.encoder(x)
        T = T.ravel()  ## odeint requires 1-D Tensor for time
        epsilon = torch.randn(qz_mean.size()).to(T.device)
        z = epsilon * torch.exp(.5 * qz_logvar) + qz_mean

        index = torch.argsort(T)
        T = T[index]
        x = x[index]
        z = z[index]
        y = y[index]
        index2 = (T[:-1] != T[1:])
        index2 = torch.cat((index2, torch.tensor([True]).to(index2.device))) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
        T = T[index2]
        x = x[index2]
        z = z[index2]
        y = y[index2]

        ## infer the latent space through ODE solver based on z0, t, and LatentODEfunc
        z0 = z[0]
        options = get_step_size(self.step_size, T[0], T[-1], len(T))
        pred_z = odeint(self.lode_func, z0, T, method = self.ode_method, options = options).view(-1, self.n_latent)

        ## reconstruct the input through Decoder and compute reconstruction loss
        if self.loss_mode == 'mse':
            pred_x1 = self.decoder(z) ## decode through latent space returned by Encoder
            pred_x2 = self.decoder(pred_z) ## decode through latent space returned by ODE solver
            recon_loss_ec = F.mse_loss(x, pred_x1, reduction='none').sum(-1).mean()
            recon_loss_ode = F.mse_loss(x, pred_x2, reduction='none').sum(-1).mean()
        elif self.loss_mode == 'nb':
            pred_x1 = self.decoder(z) ## decode through latent space returned by Encoder
            pred_x2 = self.decoder(pred_z) ## decode through latent space returned by ODE solver
            y = y.unsqueeze(1).expand(pred_x1.size(0), pred_x1.size(1))
            pred_x1 = pred_x1 * y
            pred_x2 = pred_x2 * y
            disp = torch.exp(self.decoder.disp)
            recon_loss_ec = -log_nb(x, pred_x1, disp).sum(-1).mean()
            recon_loss_ode = -log_nb(x, pred_x2, disp).sum(-1).mean()
        elif self.loss_mode == 'zinb':
            pred_x1, dp1 = self.decoder(z)
            pred_x2, dp2 = self.decoder(pred_z)
            y = y.unsqueeze(1).expand(pred_x1.size(0), pred_x1.size(1))
            pred_x1 = pred_x1 * y
            pred_x2 = pred_x2 * y
            disp = torch.exp(self.decoder.disp)
            recon_loss_ec = -log_zinb(x, pred_x1, disp, dp1).sum(-1).mean()
            recon_loss_ode = -log_zinb(x, pred_x2, disp, dp2).sum(-1).mean()
        else:
            raise ValueError(f"Unrecognized `'loss_mode'` value, should be one of `'mse'`, `'nb'` and `'zinb'`.")

        ## compute KL divergence and z divergence
        z_div = F.mse_loss(z, pred_z, reduction='none').sum(-1).mean()
        pz_mean = pz_logvar = torch.zeros_like(qz_mean)
        kl_div = normal_kl(qz_mean, qz_logvar, pz_mean, pz_logvar).sum(-1).mean()

        loss = self.alpha_recon_lec * recon_loss_ec + self.alpha_recon_lode * recon_loss_ode + z_div + self.alpha_kl * kl_div

        return loss, recon_loss_ec, recon_loss_ode, kl_div, z_div
