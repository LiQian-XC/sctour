
# scTour

<img src="https://github.com/LiQian-XC/sctour/blob/ae9b45e69941bcabf3ad498dde781eb991168b83/docs/source/_static/img/scTour_head_image.png" width="400px" align="left">

scTour is an innovative and comprehensive method for dissecting cellular dynamics by analysing datasets derived from single-cell genomics.

It provides a unifying framework to depict the full picture of developmental processes from multiple angles including developmental pseudotime, vector field and latent space, and further generalises these functionalities to a multi-task architecture for within-dataset inference and cross-dataset prediction of cellular dynamics in a batch-insensitive manner.

## Key features

- cell pseudotime estimation with no need for specifying starting cells
- transcriptomic vector field inference with no discrimination between spliced and unspliced mRNAs
- cell trajectory reconstruction using latent space that incorporates both intrinsic transcriptomic structure and extrinsic pseudotime ordering
- model-based prediction of pseudotime, vector field, and latent space for query cells/datasets/time intervals

- insensitive to batch effects 
- robust to cell subsampling
- scalable to large datasets

## Installation

[![Python Versions](https://img.shields.io/badge/python-3.7+-brightgreen.svg)](https://pypi.org/project/sctour)

```console
pip install sctour
```

## Documentation

[![Documentation Status](https://readthedocs.org/projects/sctour/badge/?version=latest)](https://sctour.readthedocs.io/en/latest/?badge=latest)

Full documentation can be found [here](https://sctour.readthedocs.io/en/latest/).

## Reference

[Qian Li, A deep learning architecture for robust inference and accurate prediction of cellular dynamics. bioRxiv, 2022](https://www.biorxiv.org/content/10.1101/2022.04.17.488600v1)
