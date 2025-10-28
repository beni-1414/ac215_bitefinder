from setuptools import setup, find_packages

setup(
    name='bitefinder-vlmodel-trainer',
    version='0.1.0',
    description='Vision-language model training for bug bite classification',
    packages=find_packages(),
    install_requires=[
        'argparse>=1.4.0',
        'google-cloud-storage>=3.4.1',
        'google-cloud-secret-manager>=2.0.0',
        'matplotlib>=3.10.7',
        'numpy>=2.3.3',
        'pandas>=2.3.3',
        'scikit-learn>=1.7.2',
        'torch>=2.8.0',
        'torchvision>=0.23.0',
        'tqdm>=4.67.1',
        'transformers>=4.57.0',
        'wandb>=0.22.2',
        'pillow',
    ],
    python_requires='>=3.12',
)
