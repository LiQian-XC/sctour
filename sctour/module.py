import torch
import torch.nn as nn
from typing_extensions import Literal


class LatentODEfunc(nn.Module):
    """
    A class modelling the latent state derivatives with respect to time.

    Parameters
    ----------
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 25)
    """

    def __init__(
        self,
        n_latent: int = 5,
        n_hidden: int = 25,
    ):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """
        Compute the gradient at a given time t and a given state x.

        Parameters
        ----------
        t
            A given time point.
        x
            A given latent state.

        Returns
        ----------
        :class:`torch.Tensor`
            A tensor
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class Encoder(nn.Module):
    """
    Encoder class generating the time and latent space.

    Parameters
    ----------
    n_int
        The dimensionality of the input.
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 5,
        n_hidden: int = 128,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_int, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc.add_module('A1', nn.ReLU())
        self.fc2 = nn.Linear(n_hidden, n_latent*2)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x:torch.Tensor) -> tuple:
        x = self.fc(x)
        out = self.fc2(x)
        qz_mean, qz_logvar = out[:, :self.n_latent], out[:, self.n_latent:]
        t = self.fc3(x).sigmoid()
        return t, qz_mean, qz_logvar


class Decoder(nn.Module):
    """
    Decoder class to reconstruct the original input based on its latent space.

    Parameters
    ----------
    n_latent
        The dimensionality of the latent space.
        (Default: 5)
    n_int
        The dimensionality of the original input.
    n_hidden
        The dimensionality of the hidden layer.
        (Default: 128)
    batch_norm
        Whether to include `BatchNorm` layer or not.
        (Default: `False`)
    loss_mode
        The mode for reconstructing the original data.
        (Default: `'nb'`)
    """

    def __init__(
        self,
        n_int: int,
        n_latent: int = 5,
        n_hidden: int = 128,
        batch_norm: bool = False,
        loss_mode: Literal['mse', 'nb', 'zinb'] = 'mse',
    ):
        super().__init__()
        self.loss_mode = loss_mode
        if loss_mode in ['nb', 'zinb']:
            self.disp = nn.Parameter(torch.randn(n_int))

        self.fc = nn.Sequential()
        self.fc.add_module('L1', nn.Linear(n_latent, n_hidden))
        if batch_norm:
            self.fc.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc.add_module('A1', nn.ReLU())

        if loss_mode == 'mse':
            self.fc2 = nn.Linear(n_hidden, n_int)
        if loss_mode in ['nb', 'zinb']:
            self.fc2 = nn.Sequential(nn.Linear(n_hidden, n_int), nn.Softmax(dim = -1))
        if loss_mode == 'zinb':
            self.fc3 = nn.Linear(n_hidden, n_int)

    def forward(self, z: torch.Tensor):
        out = self.fc(z)
        recon_x = self.fc2(out)
        if self.loss_mode == 'zinb':
            disp = self.fc3(out)
            return recon_x, disp
        else:
            return recon_x
