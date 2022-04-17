import setuptools

setuptools.setup(
    name='sctour',
    version='0.1.0',
    author='Qian Li',
    author_email='liqian.picb@gmail.com',
    description='a deep learning architecture for robust inference and accurate prediction of cellular dynamics',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.9.1',
        'torchdiffeq>=0.2.2',
        'numpy>=1.19.2',
        'scanpy>=1.7.1',
        'anndata>=0.7.5',
        'scipy>=1.5.2',
        'tqdm>=4.32.2',
        'sklearn>=1.0.1',
        'collections',
    ]
)
