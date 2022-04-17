
# About scTour

scTour is an innovative and comprehensive method for dissecting cellular dynamics by analysing datasets derived from single-cell genomics. It provides a unifying framework to depict the full picture of developmental processes from multiple angles including developmental pseudotime, vector field and latent space, and further generalises these functionalities to a multi-task architecture for within-dataset inference and cross-dataset prediction of cellular dynamics in a batch-insensitive manner.

<p align="center"><img src="https://github.com/LiQian-XC/sctour/blob/ae9b45e69941bcabf3ad498dde781eb991168b83/docs/source/_static/img/scTour_head_image.png" width="700px" align="center"></p>

# Preprint

Coming soon.

# scTour features

- unsupervised estimates of cell pseudotime along the trajectory with no need for specifying starting cells
- efficient inference of vector field with no dependence on the discrimination between spliced and unspliced mRNAs
- cell trajectory reconstruction using latent space that incorporates both intrinsic transcriptome and extrinsic time information
- model-based prediction of pseudotime, vector field, and latent space for query cells/datasets
- reconstruction of transcriptomic space given an unobserved time interval

# scTour performance

✅ insensitive to batch effects  

✅ robust to cell subsampling  

✅ scalable to large datasets

# Installation

```console
pip install sctour
```

# Documentation

[![Documentation Status](https://readthedocs.org/projects/sctour/badge/?version=latest)](https://sctour.readthedocs.io/en/latest/?badge=latest)

Full documentation can be found [here](https://sctour.readthedocs.io/en/latest/index.html#).
