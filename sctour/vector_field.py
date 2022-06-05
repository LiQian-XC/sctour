import numpy as np
import scanpy as sc
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from anndata import AnnData
from typing import Optional, Union

from ._utils import l2_norm


def cosine_similarity(
    adata: AnnData,
    zs_key: str,
    reverse: bool = False,
    use_rep_neigh: Optional[str] = None,
    vf_key: str = 'VF',
    run_neigh: bool = True,
    n_neigh: int = 20,
    t_key: Optional[str] = None,
    var_stabilize_transform: bool = False,
) -> csr_matrix:
    """
    Calculate the cosine similarity between the vector field and the cell-neighbor latent state difference for each cell.
    The calculation borrows the ideas from scvelo: https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_graph.py.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    reverse
        Whether to reverse the direction of vector field.
        (Default: `False`)
    zs_key
        The key in `.obsm` for storing the latent space.
    vf_key
        The key in `.obsm` for storing the vector field.
        (Default: `'VF'`)
    run_neigh
        Whether to run neighbor detection.
        (Default: `True`)
    use_rep_neigh
        The representation in `.obsm` for neighbor detection.
    n_neigh
        The number of neighbors for each cell.
        (Default: 20)
    t_key:
        The key in `.obs` for estimated time for neighbor detection.
    var_stabilize_transform
        Whether to perform variance-stabilizing transformation for vector field and cell-neighbor latent state difference.
        (Default: `False`)

    Returns
    ----------
    :class:`~scipy.sparse.csr_matrix`
        A sparse matrix with cosine similarities.
    """

    Z = adata.obsm[f'X_{zs_key}']
    V = adata.obsm[f'X_{vf_key}']
    if reverse:
        V = -V
    if var_stabilize_transform:
        V = np.sqrt(np.abs(V)) * np.sign(V)

    ncells = adata.n_obs

    if run_neigh:
        sc.pp.neighbors(adata, use_rep = f'X_{use_rep_neigh}', n_neighbors = n_neigh)
    n_neigh = adata.uns['neighbors']['params']['n_neighbors'] - 1
#    indices_matrix = adata.obsp['distances'].indices.reshape(-1, n_neigh)

    if t_key is not None:
        ts = adata.obs[t_key].values
        indices_matrix2 = np.zeros((ncells, n_neigh), dtype = int)
        for i in range(ncells):
            idx = np.abs(ts - ts[i]).argsort()[:(n_neigh + 1)]
            idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
            indices_matrix2[i] = idx
#        indices_matrix = np.hstack([indices_matrix, indices_matrix2])

    vals, rows, cols = [], [], []
    for i in range(ncells):
#        idx = np.unique(indices_matrix[i])
#        idx2 = indices_matrix[idx].flatten()
#        idx2 = np.setdiff1d(idx2, i)
        idx = adata.obsp['distances'][i].indices
        idx2 = adata.obsp['distances'][idx].indices
        idx2 = np.setdiff1d(idx2, i)
        idx = np.unique(np.concatenate([idx, idx2])) if t_key is None else np.unique(np.concatenate([idx, idx2, indices_matrix2[i]]))
        dZ = Z[idx] - Z[i, None]
        if var_stabilize_transform:
            dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
        cos_sim = np.einsum("ij, j", dZ, V[i]) / (l2_norm(dZ, axis = 1) * l2_norm(V[i]))
        vals.extend(cos_sim)
        rows.extend(np.repeat(i, len(idx)))
        cols.extend(idx)

    res = coo_matrix((vals, (rows, cols)), shape = (ncells, ncells))
    res.data = np.clip(res.data, -1, 1)
    return res.tocsr()


def quiver_autoscale(
    E: np.ndarray,
    V: np.ndarray,
):
    """
    Get the autoscaling in quiver.
    This function is from scvelo: https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_embedding.py.

    Parameters
    ----------
    E
        The embedding.
    V
        The weighted unitary displacement.

    Returns
    ----------
    The autoscaling factor.
    """

    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles = 'xy',
        scale = None,
        scale_units = 'xy',
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


def vector_field_embedding(
    adata: AnnData,
    T_key: str,
    E_key: str,
    scale: int = 10,
    self_transition: bool = False,
):
    """
    Calculate the weighted unitary displacement vectors under a certain embedding.
    This function borrows the ideas from scvelo: https://github.com/theislab/scvelo/blob/master/scvelo/tools/velocity_embedding.py.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    T_key
        The key in `.obsp` for cosine similarity.
    E_key
        The key in `.obsm` for embedding.
    scale
        Scale factor for cosine similarity.
        (Default: 10)
    self_transition
        Whether to take self-transition into consideration.
        (Default: `False`)

    Returns
    ----------
    The weighted unitary displacement vectors.
    """

    T = adata.obsp[T_key]

    if self_transition:
        max_t = T.max(1).A.flatten()
        ub = np.percentile(max_t, 98)
        self_t = np.clip(ub - max_t, 0, 1)
        T.setdiag(self_t)

    T = T.sign().multiply(np.expm1(abs(T * scale)))
    T = T.multiply(csr_matrix(1.0 / abs(T).sum(1)))
    if self_transition:
        T.setdiag(0)
        T.eliminate_zeros()

    E = adata.obsm[f'X_{E_key}']
    V = np.zeros(E.shape)

    for i in range(adata.n_obs):
        idx = T[i].indices
        dE = E[idx] - E[i, None]
        dE /= l2_norm(dE)[:, None]
        dE[np.isnan(dE)] = 0
        prob = T[i].data
        V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

    V /= 3 * quiver_autoscale(E, V)
    return V


def vector_field_embedding_grid(
    E: np.ndarray,
    V: np.ndarray,
    smooth: float = 0.5,
    stream: bool = False,
    density: float = 1.0,
) -> tuple:
    """
    Estimate the unitary displacement vectors within a grid.
    This function borrows the ideas from scvelo: https://github.com/theislab/scvelo/blob/master/scvelo/plotting/velocity_embedding_grid.py.

    Parameters
    ----------
    E
        The embedding.
    V
        The unitary displacement vectors under the embedding.
    smooth
        The factor for scale in Gaussian kernel.
        (Default: 0.5)
    stream
        Whether to adjust for streamplot.
        (Default: `False`)
    density
        grid density
        (Default: 1.0)

    Returns
    ----------
    tuple
        The embedding and unitary displacement vectors in grid level.
    """

    grs = []
    for i in range(E.shape[1]):
        m, M = np.min(E[:, i]), np.max(E[:, i])
        diff = M - m
        m = m - 0.01 * diff
        M = M + 0.01 * diff
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes = np.meshgrid(*grs)
    E_grid = np.vstack([i.flat for i in meshes]).T

    n_neigh = int(E.shape[0] / 50)
    nn = NearestNeighbors(n_neighbors = n_neigh, n_jobs = -1)
    nn.fit(E)
    dists, neighs = nn.kneighbors(E_grid)

    scale = np.mean([g[1] - g[0] for g in grs]) * smooth
    weight = norm.pdf(x = dists, scale = scale)
    weight_sum = weight.sum(1)

    V_grid = (V[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, weight_sum)[:, None]

    if stream:
        E_grid = np.stack(grs)
        ns = int(50 * density)
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid * V_grid).sum(0))
        min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
        cutoff1 = (mass < min_mass)

        length = np.sum(np.mean(np.abs(V[neighs]), axis = 1), axis = 1).reshape(ns, ns)
        cutoff2 = (length < np.percentile(length, 5))

        cutoff = (cutoff1 | cutoff2)
        V_grid[0][cutoff] = np.nan
    else:
        min_weight = np.percentile(weight_sum, 99) * 0.01
        E_grid, V_grid = E_grid[weight_sum > min_weight], V_grid[weight_sum > min_weight]
        V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

    return E_grid, V_grid


def plot_vector_field(
    adata: AnnData,
    reverse: bool = False,
    zs_key: Optional[str] = None,
    vf_key: str = 'VF',
    run_neigh: bool = True,
    use_rep_neigh: Optional[str] = None,
    t_key: Optional[str] = None,
    n_neigh: int = 20,
    var_stabilize_transform: bool = False,
    E_key: str = 'umap',
    scale: int = 10,
    self_transition: bool = False,
    smooth: float = 0.5,
    grid: bool = False,
    stream: bool = True,
    stream_density: int = 2,
    stream_color: str = 'k',
    linewidth: int = 1,
    arrowsize: int = 1,
    density: float = 1.,
    arrow_size_grid: int = 1,
    arrow_length_grid: int = 1,
    arrow_color_grid: str = 'grey',
    grid_density: float = 1.,
    color: Optional[str] = None,
#    ax: Optional[Axes] = None,
    **kwargs,
):
    """
    Visualize the vector field.
    The visulization of vector field under an embedding borrows the ideas from scvelo: https://github.com/theislab/scvelo.

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` object.
    reverse
        Whether to reverse the direction of vector field.
    zs_key
        The key in `.obsm` for storing the latent space.
    vf_key
        The key in `.obsm` for storing the vector field.
    run_neigh
        Whether to run neighbor detection.
    use_rep_neigh
        The representation in `.obsm` for neighbor detection.
    t_key:
        The key in `.obs` for estimated time for neighbor detection.
    n_neigh
        The number of neighbors for each cell.
    var_stabilize_transform
        Whether to perform variance-stabilizing transformation for vector field and cell-neighbor latent state difference.
    E_key
        The key in `.obsm` for embedding.
    scale
        Scale factor for cosine similarity.
    self_transition
        Whether to take self-transition into consideration.
    smooth
        The factor for scale in Gaussian kernel.
    grid
        Whether to draw grid-level vector field.
    stream
        Whether to draw streamplot.
    stream_density
        The density parameter for streamplot for controlling the closeness of the streamlines.
    stream_color
        The streamline color for streamplot.
    linewidth
        The line width for streamplot.
    arrowsize
        The arrow size for streamplot.
    density
        Percentage of cell positions to show.
    arrow_size_grid
        The arrow size in grid-level vector field.
    arrow_length_grid
        The arrow length in grid-level vector field
    arrow_color_grid
        The arrow color in grid-level vector field
    grid_density
        The grid-level density for showing vector field
    color
        `color` parameter in :func:`scanpy.pl.umap`.
    ax
        The matplotlib axes
    kwargs
        Parameters passed to :func:`scanpy.pl.umap`

    Returns
    ----------
    :class:`~matplotlib.axes.Axes`
        An :class:`~matplotlib.axes.Axes` object.
    """

    ##calculate cosine similarity
    adata.obsp['cosine_similarity'] = cosine_similarity(
                                        adata,
                                        reverse = reverse,
                                        zs_key = zs_key,
                                        vf_key = vf_key,
                                        run_neigh = run_neigh,
                                        use_rep_neigh = use_rep_neigh,
                                        t_key = t_key,
                                        n_neigh = n_neigh,
                                        var_stabilize_transform = var_stabilize_transform,
                                        )
    ##get weighted unitary displacement vectors under a certain embedding
    adata.obsm['X_DV'] = vector_field_embedding(
                            adata,
                            T_key = 'cosine_similarity',
                            E_key = E_key,
                            scale = scale,
                            self_transition = self_transition,
                        )

    E = adata.obsm[f'X_{E_key}']
    V = adata.obsm[f'X_DV']

    if grid:
        stream = False

    if grid or stream:
        E, V = vector_field_embedding_grid(
                E = E,
                V = V,
                smooth = smooth,
                stream = stream,
                density = grid_density,
                )

    ax = sc.pl.embedding(adata, basis = E_key, color = color, show=False, **kwargs)
    if stream:
        lengths = np.sqrt((V * V).sum(0))
        linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
        stream_kwargs = dict(
            linewidth = linewidth,
            density = stream_density,
            zorder = 3,
            color = stream_color,
            arrowsize = arrowsize,
            arrowstyle = '-|>',
            maxlength = 4,
            integration_direction = 'both',
        )
        ax.streamplot(E[0], E[1], V[0], V[1], **stream_kwargs)
    else:
        if density < 1:
            idx = np.random.choice(len(E), int(len(E) * density), replace = False)
            E = E[idx]
            V = V[idx]
        scale = 1 / arrow_length_grid
        hl, hw, hal = 6 * arrow_size_grid, 5 * arrow_size_grid, 4 * arrow_size_grid
        quiver_kwargs = dict(
            angles = 'xy',
            scale_units = 'xy',
            edgecolors = 'k',
            scale = scale,
            width = 0.001,
            headlength = hl,
            headwidth = hw,
            headaxislength = hal,
            color = arrow_color_grid,
            linewidth = 0.2,
            zorder = 3,
        )
        ax.quiver(E[:, 0], E[:, 1], V[:, 0], V[:, 1], **quiver_kwargs)

#    ax = sc.pl.embedding(adata, basis = E_key, color = color, ax = ax, show = False, **kwargs)

    return ax
