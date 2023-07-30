import torch
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from typing import Optional, Union
from typing_extensions import Literal
import numpy as np
from anndata import AnnData
from scipy import sparse
from scipy.sparse import spmatrix
from tqdm import tqdm
import os
from collections import defaultdict

from .model import TNODE
from ._utils import get_step_size
from .data import split_data, MakeDataset, BatchSampler
from . import logger


##reverse time
def reverse_time(
    T: np.ndarray,
) -> np.ndarray:
    """
    Post-inference adjustment to reverse the pseudotime.

    Parameters
    ----------
    T
        The pseudotime inferred for each cell.

    Returns
    ----------
    :class:`~numpy.ndarray`
        The reversed pseudotime.
    """

    return 1 - T


class Trainer:
    """
    Class for implementing the scTour training process.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object for the training data.
    percent
        The percentage of cells used for model training. Default to 0.2 when the cell number > 10,000 and to 0.9 otherwise.
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
        Whether to include a `BatchNorm` layer.
        (Default: `False`)
    ode_method
        The solver for ODE. List of ODE solvers can be found in `torchdiffeq`.
        (Default: `'euler'`)
    step_size
        The step size during integration.
    alpha_recon_lec
        The scaling factor for the reconstruction error from encoder-derived latent space.
        (Default: 0.5)
    alpha_recon_lode
        The scaling factor for the reconstruction error from ODE-solver-derived latent space.
        (Default: 0.5)
    alpha_kl
        The scaling factor for the KL divergence in the loss function.
        (Default: 1.0)
    loss_mode
        The mode for calculating the reconstruction error.
        (Default: `'nb'`)
        Three modes are included:
        ``'mse'``: mean squared error;
        ``'nb'``: negative binomial conditioned likelihood;
        ``'zinb'``: zero-inflated negative binomial conditioned likelihood.
    nepoch
        Number of epochs.
    batch_size
        The batch size during training.
        (Default: 1024)
    drop_last
        Whether or not drop the last batch when its size is smaller than `batch_size`.
        (Default: `False`)
    lr
        The learning rate.
        (Default: 1e-3)
    wt_decay
        The weight decay (L2 penalty) for Adam optimizer.
        (Default: 1e-6)
    eps
        The `eps` parameter for Adam optimizer.
        (Default: 0.01)
    random_state
        The seed for generating random numbers.
        (Default: 0)
    val_frac
        The percentage of data used for validation.
        (Default: 0.1)
    use_gpu
        Whether to use GPU when available.
        (Default: `True`)
    """

    def __init__(
        self,
        adata: AnnData,
        percent: Optional[float] = None,
        n_latent: int = 5,
        n_ode_hidden: int = 25,
        n_vae_hidden: int = 128,
        batch_norm: bool = False,
        ode_method: str = 'euler',
        step_size: Optional[int] = None,
        alpha_recon_lec: float = 0.5,
        alpha_recon_lode: float = 0.5,
        alpha_kl: float = 1.,
        loss_mode: Literal['mse', 'nb', 'zinb'] = 'nb',
        nepoch: Optional[int] = None,
        batch_size: int = 1024,
        drop_last: bool = False,
        lr: float = 1e-3,
        wt_decay: float = 1e-6,
        eps: float = 0.01,
        random_state: int = 0,
        val_frac: float = 0.1,
        use_gpu: bool = True,
    ):
        self.loss_mode = loss_mode
        if self.loss_mode not in ['mse', 'nb', 'zinb']:
            raise ValueError(
                    f"`loss_mode` must be one of ['mse', 'nb', 'zinb'], but input was '{self.loss_mode}'."
                    )

        if (alpha_recon_lec < 0) or (alpha_recon_lec > 1):
            raise ValueError(
                    '`alpha_recon_lec` must be between 0 and 1.'
                    )
        if (alpha_recon_lode < 0) or (alpha_recon_lode > 1):
            raise ValueError(
                    '`alpha_recon_lode` must be between 0 and 1.'
                    )
        if alpha_recon_lec + alpha_recon_lode != 1:
            raise ValueError(
                    'The sum of `alpha_recon_lec` and `alpha_recon_lode` must be 1.'
                    )

        self.adata = adata
        if 'n_genes_by_counts' not in self.adata.obs:
            raise KeyError(
                    "`n_genes_by_counts` not found in `.obs` of the AnnData. Please run `scanpy.pp.calculate_qc_metrics` first to calculate the number of genes detected in each cell."
                    )
        if loss_mode == 'mse':
            if (self.adata.X.min() < 0) or (self.adata.X.max() > np.log1p(1e6)):
                raise ValueError(
                        "Invalid expression matrix in `.X`. `mse` mode expects log1p(normalized expression) in `.X` of the AnnData."
                        )
        else:
            X = self.adata.X.data if sparse.issparse(self.adata.X) else self.adata.X
            if (X.min() < 0) or np.any(~np.equal(np.mod(X, 1), 0)):
                raise ValueError(
                        f"Invalid expression matrix in `.X`. `{self.loss_mode}` mode expects raw UMI counts in `.X` of the AnnData."
                        )

        self.n_cells = adata.n_obs
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.percent = percent
        if self.percent is None:
            if self.n_cells > 10000:
                self.percent = .2
            else:
                self.percent = .9
        else:
            if (self.percent < 0) or (self.percent > 1):
                raise ValueError(
                        "`percent` must be between 0 and 1."
                        )
        self.val_frac = val_frac
        if (self.val_frac < 0) or (self.val_frac > 1):
            raise ValueError(
                    '`val_frac` must be between 0 and 1.'
                    )

        if nepoch is None:
            ncells = round(self.n_cells * self.percent)
            self.nepoch = np.min([round((10000 / ncells) * 400), 400])
        else:
            self.nepoch = nepoch

        self.lr = lr
        self.wt_decay = wt_decay
        self.eps = eps
        self.time_reverse = None

        self.random_state = random_state
        np.random.seed(random_state)
#       random.seed(random_state)
        torch.manual_seed(random_state)
#       torch.backends.cudnn.benchmark = False
#       torch.use_deterministic_algorithms(True)

        self.use_gpu = use_gpu
        gpu = torch.cuda.is_available() and use_gpu
        if gpu:
            torch.cuda.manual_seed(random_state)
            self.device = torch.device('cuda')
            logger.info('Running using GPU.')
        else:
            self.device = torch.device('cpu')
            logger.info('Running using CPU.')

        self.n_int = adata.n_vars
        self.model_kwargs = dict(
            device = self.device,
            n_int = self.n_int,
            n_latent = n_latent,
            n_ode_hidden = n_ode_hidden,
            n_vae_hidden = n_vae_hidden,
            batch_norm = batch_norm,
            ode_method = ode_method,
            step_size = step_size,
            alpha_recon_lec = alpha_recon_lec,
            alpha_recon_lode = alpha_recon_lode,
            alpha_kl = alpha_kl,
            loss_mode = loss_mode,
        )
        self.model = TNODE(**self.model_kwargs)
        self.log = defaultdict(list)


    def _get_data_loaders(self) -> None:
        """
        Generate Data Loaders for training and validation datasets.
        """

        train_data, val_data = split_data(self.adata, self.percent, self.val_frac)
        self.train_dataset = MakeDataset(train_data, self.loss_mode)
        self.val_dataset = MakeDataset(val_data, self.loss_mode)

#        sampler = BatchSampler(train_data.n_obs, self.batch_size, self.drop_last)
#        self.train_dl = DataLoader(self.train_dataset, batch_sampler = sampler)
        self.train_dl = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_dl = DataLoader(self.val_dataset, batch_size = self.batch_size)


    def train(self):
        """
        Model training.
        """
        self._get_data_loaders()

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr = self.lr, weight_decay = self.wt_decay, eps = self.eps)

        with tqdm(total=self.nepoch, unit='epoch') as t:
            for tepoch in range(t.total):
                train_loss = self._on_epoch_train(self.train_dl)
                val_loss = self._on_epoch_val(self.val_dl)
                self.log['train_loss'].append(train_loss)
                self.log['validation_loss'].append(val_loss)
                t.set_description(f"Epoch {tepoch + 1}")
                t.set_postfix({'train_loss': train_loss, 'val_loss': val_loss}, refresh=False)
                t.update()


    def _on_epoch_train(self, DL) -> float:
        """
        Go through the model and update the model parameters.

        Parameters
        ----------
        DL
            DataLoader for training dataset.

        Returns
        ----------
        float
            Training loss for the current epoch.
        """

        self.model.train()
        total_loss = .0
        ss = 0
        for X, Y in DL:
            self.optimizer.zero_grad()
            X = X.to(self.device)
            Y = Y.to(self.device)
            loss, recon_loss_ec, recon_loss_ode, kl_div, z_div = self.model(X, Y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            ss += X.size(0)

        train_loss = total_loss/ss
        return train_loss


    @torch.no_grad()
    def _on_epoch_val(self, DL) -> float:
        """
        Validate using validation dataset.

        Parameters
        ----------
        DL
            DataLoader for validation dataset.

        Returns
        ----------
        float
            Validation loss for the current epoch.
        """

        self.model.eval()
        total_loss = .0
        ss = 0
        for X, Y in DL:
            X = X.to(self.device)
            Y = Y.to(self.device)
            loss, recon_loss_ec, recon_loss_ode, kl_div, z_div = self.model(X, Y)
            total_loss += loss.item() * X.size(0)
            ss += X.size(0)

        val_loss = total_loss/ss
        return val_loss


    def get_time(
        self,
        ) -> np.ndarray:
        """
        Infer the developmental pseudotime.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The pseudotime inferred for each cell.
        """

        X = self.adata.X
        if self.loss_mode in ['nb', 'zinb']:
            X = np.log1p(X)
        ts = self._get_time(self.model, X)

        ## The model might return pseudotime in reverse order. Check this based on number of genes expressed in each cell.
        if self.time_reverse is None:
            n_genes = torch.tensor(self.adata.obs['n_genes_by_counts'].values).float().log1p().to(self.device)
            m_ts = ts.mean()
            m_ngenes = n_genes.mean()
            beta_direction = (ts * n_genes).sum() - len(ts) * m_ts * m_ngenes
            if beta_direction > 0:
                self.time_reverse = True
            else:
                self.time_reverse = False
        if self.time_reverse:
            ts = 1 - ts

        return ts.cpu().numpy()


    def get_vector_field(
        self,
        T: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Infer the vector field.

        Parameters
        ----------
        T
            The pseudotime estimated for each cell.
        Z
            The latent representation for each cell.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The estimated vector field.
        """

        vf = self._get_vector_field(
                                    self.model,
                                    T,
                                    Z,
                                    self.time_reverse,
                                    )
        return vf


    def get_latentsp(
        self,
        alpha_z: float = .5,
        alpha_predz: float = .5,
        step_size: Optional[int] = None,
        step_wise: bool = False,
        batch_size: Optional[int] = None,
    ) -> tuple:
        """
        Infer the latent space.

        Parameters
        ----------
        alpha_z
            Scaling factor for encoder-derived latent space.
            (Default: 0.5)
        alpha_predz
            Scaling factor for ODE-solver-derived latent space.
            (Default: 0.5)
        step_size
            Step size during integration.
        step_wise
            Whether to perform step-wise integration by iteratively considering only two time points each time.
            (Default: `False`)
        batch_size
            Batch size when deriving the latent space. The default is no mini-batching.

        Returns
        ----------
        tuple
            3-tuple of weighted combined latent space, encoder-derived latent space, and ODE-solver-derived latent space.
        """

        X = self.adata.X
        if self.model.loss_mode in ['nb', 'zinb']:
            X = np.log1p(X)
        mix_zs, zs, pred_zs = self._get_latentsp(self.model,
                                                 X,
                                                 alpha_z,
                                                 alpha_predz,
                                                 step_size,
                                                 step_wise,
                                                 batch_size,
                                                 )
        return mix_zs, zs, pred_zs


    def save_model(
        self,
        save_dir: str,
        save_prefix: str,
    ) -> None:
        """
        Save the trained scTour model.

        Parameters
        ----------
        save_dir
            The directory where the model will be saved.
        save_prefix
            The prefix for model name. The model will be saved in 'save_dir/save_prefix.pth'.
        """

        save_path = os.path.abspath(os.path.join(save_dir, f'{save_prefix}.pth'))
#        save_path = os.path.abspath(os.path.join(save_dir, f'{save_prefix}.tar'))
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_kwargs': self.model_kwargs,
                'time_reverse': self.time_reverse,
                'adata': self.adata,
                'percent': self.percent,
                'nepoch': self.nepoch,
                'batch_size': self.batch_size,
                'random_state': self.random_state,
                'drop_last': self.drop_last,
                'lr': self.lr,
                'wt_decay': self.wt_decay,
                'eps': self.eps,
                'val_frac': self.val_frac,
                'use_gpu': self.use_gpu,
            },
            save_path
        )


    @staticmethod
    @torch.no_grad()
    def _get_time(
        model: TNODE,
        X: Union[np.ndarray, spmatrix],
        ) -> torch.tensor:
        """
        Derive the developmental pseudotime for cells.

        Parameters
        ----------
        model
            The trained scTour model.
        X
            The data matrix.

        Returns
        ----------
        :class:`torch.Tensor`
            The pseudotime estimated for each cell.
        """

        model.eval()
        if sparse.issparse(X):
            X = X.A
        X = torch.tensor(X).to(model.device)
        ts, _, _ = model.encoder(X)
        ts = ts.ravel()
        return ts


    @staticmethod
    @torch.no_grad()
    def _get_vector_field(
        model: TNODE,
        T: np.ndarray,
        Z: np.ndarray,
        time_reverse: bool,
    ) -> np.ndarray:
        """
        Derive the vector field for cells.

        Parameters
        ----------
        model
            The trained scTour model.
        T
            The pseudotime for each cell.
        Z
            The latent representation for each cell.
        time_reverse
            Whether to reverse the vector field.

        Returns
        ----------
        :class:`~numpy.ndarray`
            The estimated vector field.
        """

        model.eval()
        if not (isinstance(T, np.ndarray) and isinstance(Z, np.ndarray)):
            raise TypeError(
                    'The inputs must be numpy arrays.'
                    )
        Z = torch.tensor(Z)
        T = torch.tensor(T)
        if time_reverse is None:
            raise RuntimeError(
                    'It seems you did not run `get_time()` function first after model training.'
                    )
        direction = 1
        if time_reverse:
            direction = -1
        return direction * model.lode_func(T, Z).numpy()


    @staticmethod
    @torch.no_grad()
    def _get_latentsp(
        model: TNODE,
        X: Union[np.ndarray, spmatrix],
        alpha_z: float = .5,
        alpha_predz: float = .5,
        step_size: Optional[int] = None,
        step_wise: bool = False,
        batch_size: Optional[int] = None,
    ):
        """
        Derive the latent representations of cells.

        Parameters
        ----------
        model
            The trained scTour model.
        X
            The data matrix.
        alpha_z
            Scaling factor for encoder-derived latent space.
            (Default: 0.5)
        alpha_predz
            Scaling factor for ODE-solver-derived latent space.
            (Default: 0.5)
        step_size
            Step size during integration.
        step_wise
            Whether to perform step-wise integration by iteratively considering only two time points each time.
            (Default: `False`)
        batch_size
            Batch size when deriving the latent space. The default is no mini-batching.

        Returns
        ----------
        tuple
            3-tuple of weighted combined latent space, encoder-derived latent space, and ODE-solver-derived latent space.
        """

        model.eval()

        if (alpha_z < 0) or (alpha_z > 1):
            raise ValueError(
                    '`alpha_z` must be between 0 and 1.'
                    )
        if (alpha_predz < 0) or (alpha_predz > 1):
            raise ValueError(
                    '`alpha_predz` must be between 0 and 1.'
                    )
        if alpha_z + alpha_predz != 1:
            raise ValueError(
                    'The sum of `alpha_z` and `alpha_predz` must be 1.'
                    )

        if sparse.issparse(X):
            X = X.A
        X = torch.tensor(X).to(model.device)
        T, qz_mean, qz_logvar = model.encoder(X)
        T = T.ravel().cpu()
        epsilon = torch.randn(qz_mean.size())
        zs = epsilon * torch.exp(.5 * qz_logvar.cpu()) + qz_mean.cpu()

        sort_T, sort_idx, sort_ridx = np.unique(T, return_index=True, return_inverse=True)
        sort_T = torch.tensor(sort_T)
        sort_zs = zs[sort_idx]

        pred_zs = []
        if batch_size is None:
            batch_size = len(sort_T)
        times = int(np.ceil(len(sort_T) / batch_size))
        for i in range(times):
            idx1 = i * batch_size
            idx2 = np.min([(i + 1)*batch_size, len(sort_T)])
            t = sort_T[idx1:idx2]
            z = sort_zs[idx1:idx2]
            z0 = z[0]

            if not step_wise:
                options = get_step_size(step_size, t[0], t[-1], len(t))
                pred_z = odeint(
                                model.lode_func,
                                z0,
                                t,
                                method = model.ode_method,
                                options = options
                                ).view(-1, model.n_latent)
            else:
                pred_z = torch.empty((len(t), z.size(1)))
                pred_z[0] = z0
                for j in range(len(t) - 1):
                    t2 = t[j:(j + 2)]
                    options = get_step_size(step_size, t2[0], t2[-1], len(t2))
                    pred_z[j + 1] = odeint(
                                            model.lode_func,
                                            z[j],
                                            t2,
                                            method = model.ode_method,
                                            options = options
                                    )[1]

            pred_zs += [pred_z]

        pred_zs = torch.cat(pred_zs)
        pred_zs = pred_zs[sort_ridx]
        mix_zs = alpha_z * zs + alpha_predz * pred_zs

        return mix_zs.numpy(), zs.numpy(), pred_zs.numpy()
