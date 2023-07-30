import torch
from torchdiffeq import odeint
from typing import Optional, Union
from typing_extensions import Literal
import numpy as np
from anndata import AnnData
from scipy import sparse
from scipy.sparse import spmatrix
import os

from ._utils import get_step_size
from .train import Trainer


def _check_data(
    adata1: AnnData,
    adata2: AnnData,
    loss_mode: str,
) -> np.ndarray:
    """
    Check the query data.

    Parameters
    ----------
    adata1
        An :class:`~anndata.AnnData` object for the query dataset.
    adata2
        An :class:`~anndata.AnnData` object for the training dataset.
    loss_mode
        The `loss_mode` used for model training.

    Returns
    ----------
    :class:`~numpy.ndarray`
        The expression matrix for the query dataset.
    """

    if len(adata1.var_names.intersection(adata2.var_names)) != adata2.n_vars:
        raise ValueError(
                "The query AnnData must contain all the genes that are used for training in the training dataset."
                )

    X = adata1[:, adata2.var_names].X
    if loss_mode == 'mse':
        if (X.min() < 0) or (X.max() > np.log1p(1e6)):
            raise ValueError(
                    "Invalid expression matrix in `.X`. Model trained from `mse` mode expects log1p(normalized expression) in `.X` of the query AnnData."
                    )
    else:
        data = X.data if sparse.issparse(X) else X
        if (data.min() < 0) or np.any(~np.equal(np.mod(data, 1), 0)):
            raise ValueError(
                    f"Invalid expression matrix in `.X`. Model trained from `{loss_mode}` mode expects raw UMI counts in `.X` of the query AnnData."
                    )
        else:
            X = np.log1p(X)

    return X


def load_model(model: str):
    """
    Load the trained scTour model for prediction.

    Parameters
    ----------
    model
        Filename for the scTour model trained and saved.

    Returns
    ----------
    :class:`sctour.train.Trainer`
        The trained scTour model.
    """

    if not os.path.isfile(model):
        raise FileNotFoundError(
                f'No such file: `{model}`.'
                )

    checkpoint = torch.load(model, map_location=torch.device('cpu'))
#   checkpoint = torch.load(model)
    model_kwargs = checkpoint['model_kwargs']
    del model_kwargs['device']
    del model_kwargs['n_int']
    tnode = Trainer(
            adata = checkpoint['adata'],
            percent = checkpoint['percent'],
            nepoch = checkpoint['nepoch'],
            batch_size = checkpoint['batch_size'],
            drop_last = checkpoint['drop_last'],
            lr = checkpoint['lr'],
            wt_decay = checkpoint['wt_decay'],
            eps = checkpoint['eps'],
            random_state = checkpoint['random_state'],
            val_frac = checkpoint['val_frac'],
            use_gpu = checkpoint['use_gpu'],
            **model_kwargs,
            )
    tnode.model.load_state_dict(checkpoint['model_state_dict'])
    tnode.time_reverse = checkpoint['time_reverse']
    return tnode


def predict_time(
    model: Trainer,
    adata: AnnData,
    reverse: bool = False,
) -> np.ndarray:
    """
    Predict the pseudotime for query cells.

    Parameters
    ----------
    model
        A :class:`sctour.train.Trainer` for trained scTour model.
    adata
        An :class:`~anndata.AnnData` object for the query dataset.
    reverse
        Whether to reverse the predicted pseudotime. When the pseudotime returned by `get_time()` function for the training data was in reverse order and you used the post-inference adjustment (`reverse_time()` function), please set this parameter to `True`.
        (Default: `False`)

    Returns
    ----------
    :class:`~numpy.ndarray`
        The pseudotime predicted for the query cells.
    """

    if model.time_reverse is None:
        raise RuntimeError(
                'It seems you did not run `get_time()` function after model training. Please run `get_time()` first after training for the training data before you run `predict_time()` for the query data.'
                )

    X = _check_data(adata, model.adata, model.loss_mode)
    ts = model._get_time(model = model.model, X = X)
    if model.time_reverse:
        ts = 1 - ts
    if reverse:
        ts = 1 - ts
    return ts.cpu().numpy()


def predict_vector_field(
    model: Trainer,
    T: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """
    Predict the vector field for query cells.

    Parameters
    ----------
    model
        A :class:`sctour.train.Trainer` for trained scTour model.
    T
        The predicted pseudotime for query cells.
    Z
        The predicted latent representations for query cells.

    Returns
    ----------
    :class:`~numpy.ndarray`
        The vector field predicted for query cells.
    """

    vf = model._get_vector_field(
                                model = model.model,
                                T = T,
                                Z = Z,
                                time_reverse = model.time_reverse,
                                )
    return vf


def predict_latentsp(
    model: Trainer,
    adata: AnnData,
    mode: Literal['coarse', 'fine'] = 'fine',
    alpha_z: float = .5,
    alpha_predz: float = .5,
    step_size: Optional[int] = None,
    step_wise: bool = False,
    batch_size: Optional[int] = None,
) -> tuple:
    """
    Predict the latent representations for query cells given their transcriptomes.

    Parameters
    ----------
    model
        A :class:`sctour.train.Trainer` for trained scTour model.
    adata
        An :class:`~anndata.AnnData` object for the query dataset.
    mode
        The mode for deriving the latent space for the query dataset.
        Two modes are included:
        ``'fine'``: derive the latent space by taking the training data into consideration;
        ``'coarse'``: derive the latent space directly from the query data without involving the training data.
    alpha_z
        Scaling factor for encoder-derived latent space.
        (Default: 0.5)
    alpha_predz
        Scaling factor for ODE-solver-derived latent space.
        (Default: 0.5)
    step_size
        The step size during integration.
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

    X = _check_data(adata, model.adata, model.loss_mode)
    if mode == 'coarse':
        mix_zs, zs, pred_zs = model._get_latentsp(
                                model = model.model,
                                X = X,
                                alpha_z = alpha_z,
                                alpha_predz = alpha_predz,
                                step_size = step_size,
                                step_wise = step_wise,
                                batch_size = batch_size,
                                )
    if mode == 'fine':
        X2 = model.adata.X
        if model.loss_mode in ['nb', 'zinb']:
            X2 = np.log1p(X2)
        if sparse.issparse(X2):
            X2 = X2.A
        if sparse.issparse(X):
            X = X.A
        mix_zs, zs, pred_zs = model._get_latentsp(
                                model = model.model,
                                X = np.vstack((X, X2)),
                                alpha_z = alpha_z,
                                alpha_predz = alpha_predz,
                                step_size = step_size,
                                step_wise = step_wise,
                                batch_size = batch_size,
                                )
        mix_zs = mix_zs[:len(X)]
        zs = zs[:len(X)]
        pred_zs = pred_zs[:len(X)]

    return mix_zs, zs, pred_zs


@torch.no_grad()
def predict_ltsp_from_time(
    model: Trainer,
    T: np.ndarray,
    reverse: bool = False,
    step_wise: bool = True,
    step_size: Optional[int] = None,
    alpha_z: float = 0.5,
    alpha_predz: float = 0.5,
    k: int = 20,
) -> np.ndarray:
    """
    Predict the transcriptomic latent space for query (unobserved) time intervals.

    Parameters
    ----------
    model
        A :class:`sctour.train.Trainer` for trained scTour model.
    T
        A 1D numpy array containing the query time points (with values between 0 and 1). The latent space corresponding to these time points will be predicted.
    reverse
        When the pseudotime returned by `get_time()` function for the training data was in reverse order and you used the post-inference adjustment (`reverse_time()` function), please set this parameter to `True`.
        (Default: `False`)
    step_wise
        Whether to perform step-wise integration by iteratively considering only two time points when inferring the reference latent space from the training data.
        (Default: `True`)
    step_size
        The step size during integration.
    alpha_z
        Scaling factor for encoder-derived latent space.
        (Default: 0.5)
    alpha_predz
        Scaling factor for ODE-solver-derived latent space.
        (Default: 0.5)
    k
        The k nearest neighbors in the time space considered when predicting the latent representation for each query time point.
        (Default: 20)

    Returns
    ----------
    :class:`~numpy.ndarray`
        Predicted latent space corresponding to the query time interval.
    """

    mdl = model.model

    if not isinstance(T, np.ndarray):
        raise TypeError(
                "The input time interval must be a numpy array."
                )
    if len(T.shape) > 1:
        raise TypeError(
                "The input time interval must be a 1D numpy array."
                )
    if np.any(T < 0) or np.any(T > 1):
        raise ValueError(
                "The input time points must be in [0, 1]."
                )

    ridx = np.random.permutation(len(T))
    rT = torch.tensor(T[ridx])
    ## get the reference time and latent space from the training data
    X = model.adata.X
    if model.loss_mode in ['nb', 'zinb']:
        X = np.log1p(X)
    mix_zs, zs, pred_zs = model._get_latentsp(model = mdl,
                                             X = X,
                                             alpha_z = alpha_z,
                                             alpha_predz = alpha_predz,
                                             step_wise = step_wise,
                                             step_size = step_size,
                                             )
    ts = model._get_time(model = mdl, X = X)
    if model.time_reverse:
        ts = 1 - ts
    if reverse:
        ts = 1 - ts

    ts = ts.cpu()
    zs = torch.tensor(mix_zs)

    pred_T_zs = torch.empty((len(rT), mdl.n_latent))
    for i, t in enumerate(rT):
        diff = torch.abs(t - ts)
        idxs = torch.argsort(diff)
#            n = (diff == 0).sum()
#            idxs = idxs[n:(k + n)]
        if (diff == 0).any():
            pred_T_zs[i] = zs[idxs[0]].clone()
        else:
            idxs = idxs[:k]
            k_zs = torch.empty((k, mdl.n_latent))
            for j, idx in enumerate(idxs):
                z0 = zs[idx].clone()
                t0 = ts[idx].clone()
                pred_t = torch.stack((t0, t))
                if pred_t[0] < pred_t[1]:
                    options = get_step_size(step_size, pred_t[0], pred_t[-1], len(pred_t))
                else:
                    options = get_step_size(step_size, pred_t[-1], pred_t[0], len(pred_t))
                k_zs[j] = odeint(
                                mdl.lode_func,
                                z0,
                                pred_t,
                                method = mdl.ode_method,
                                options = options
                            )[1]
            k_zs = torch.mean(k_zs, dim = 0)
            pred_T_zs[i] = k_zs
            ts = torch.cat((ts, t.unsqueeze(0)))
            zs = torch.cat((zs, k_zs.unsqueeze(0)))

    pred_T_zs = pred_T_zs[np.argsort(ridx)]
    return pred_T_zs.numpy()
