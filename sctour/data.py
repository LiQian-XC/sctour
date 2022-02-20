import torch
from torch.utils.data import Dataset
from scipy import sparse
import numpy as np
from anndata import AnnData


def split_data(
    adata: AnnData,
    percent: float,
    val_frac: float = 0.1,
):
    """
    Split the dataset for training and validation

    Parameters
    ----------
    adata
        The `AnnData` object for the whole dataset
    percent
        The percentage to be used for training the model
    val_frac
        The percentage to be used for validation

    Returns
    ----------
    `AnnData` object for training and validation
    """

    n_cells = adata.n_obs
    n_train = int(np.ceil(n_cells * percent))
    n_val = min(int(np.floor(n_train * val_frac)), n_cells - n_train)

    indices = np.random.permutation(n_cells)
    train_idx = np.random.choice(indices, n_train, replace = False)
    indices2 = np.setdiff1d(indices, train_idx)
    val_idx = np.random.choice(indices2, n_val, replace = False)
#   train_idx = indices[:n_train]
#   val_idx = indices[n_train:(n_train + n_val)]

    train_data = adata[train_idx, :]
    val_data = adata[val_idx, :]

    return train_data, val_data


def split_index(
    n_cells: int,
    percent: float,
    val_frac: float = 0.1,
):
    """
    Split the indices for training and validation

    Parameters
    ----------
    n_cells
        The total number of cells
    percent
        The percentage to be used for training the model
    val_frac
        The percentage to be used for validation

    Returns
    ----------
    2-tuple of indices
    """

    n_train = int(np.ceil(n_cells * percent))
    n_val = int(np.ceil(n_train * val_frac))
    indices = np.random.permutation(n_cells)
    train_idx = np.random.choice(indices, n_train, replace = False)
    indices2 = np.setdiff1d(indices, train_idx)
    val_idx = np.random.choice(indices2, n_val, replace = False)
#   train_idx = indices[:n_train]
#   val_idx = indices[n_train:(n_train + n_val)]
    return train_idx, val_idx


class MakeDataset(Dataset):
    """
    A class to generate Dataset

    Parameters
    ----------
    adata
        An `AnnData` object
    """

    def __init__(
        self,
        adata: AnnData,
    ):
        X = adata.X
        if sparse.issparse(X):
            X = X.A
        self.data = torch.tensor(X)
        self.library_size = self.data.sum(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx, :], self.library_size[idx]
