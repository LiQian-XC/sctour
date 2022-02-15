import setuptools

setuptools.setup(
    name='sctour',
    version='0.0.1',
    author='Qian Li',
    author_email='liqian.picb@gmail.com',
    description='a tool for robust time, transcriptome, trajectory inference',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'torchdiffeq',
        'numpy',
        'scanpy',
        'anndata',
        'scipy',
        'tqdm',
    ]
)
