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
    n_val = min(int(np.floor(n_train * val_frac)), n_cells - n_train)
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
        loss_mode: str,
    ):
        X = adata.X
        if loss_mode in ['nb', 'zinb']:
            X = np.log1p(X)
        if sparse.issparse(X):
            X = X.A
        self.data = torch.tensor(X)
        self.library_size = self.data.sum(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx, :], self.library_size[idx]


class BatchSampler():
    """
    A class to generate mini-batches through two rounds of randomization

    Parameters
    ----------
    n
        Total number of cells
    batch_size
        Size of mini-batches
    drop_last
        Whether or not drop the last batch when its size is smaller than the batch_size
    """
    def __init__(
        self,
        n: int,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.n = n
        self.drop_last = drop_last

    def __iter__(self):
        seq_n = torch.randperm(self.n)
        lb = self.n // self.batch_size
        idxs = np.arange(self.n)
        for i in range(lb):
            idx = np.random.choice(idxs, self.batch_size, replace=False)
            yield seq_n[idx].tolist()
            idxs = np.setdiff1d(idxs, idx)
        if (not self.drop_last) and (len(idxs) > 0):
            yield seq_n[idxs].tolist()

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        else:
            return int(np.ceil(self.n / self.batch_size))
