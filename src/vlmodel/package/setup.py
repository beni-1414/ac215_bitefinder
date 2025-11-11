from setuptools import setup, find_packages

setup(
    name='bitefinder-vlmodel-trainer',
    version='0.1.0',
    description='Vision-language model training for bug bite classification',
    packages=find_packages(),
    install_requires=[
        'argparse',
        'google-cloud-storage',
        'google-cloud-secret-manager',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'torchvision',
        'tqdm',
        'transformers',
        'wandb',
        'pillow',
    ],
    python_requires='>=3.10',
)
