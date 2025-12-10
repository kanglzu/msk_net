"""
Setup script for MSK-Net package
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network"

# Core dependencies (minimal set for installation)
CORE_REQUIREMENTS = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "monai>=1.3.0",
    "nibabel>=5.2.0",
    "tqdm>=4.66.0",
    "PyYAML>=6.0.0",
    "einops>=0.7.0",
    "tensorboard>=2.15.0",
]

setup(
    name="msk-net",
    version="1.0.0",
    author="Yutong Wang, Zhongfeng Kang, et al.",
    author_email="kangzf@lzu.edu.cn",
    description="MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network for 3D Brain Tumor Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kanglzu/msk_net",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "full": [
            "SimpleITK>=2.3.0",
            "MedPy>=0.4.0",
            "torchio>=0.19.0",
            "albumentations>=1.3.0",
            "wandb>=0.16.0",
            "captum>=0.6.0",
            "timm>=0.9.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
)

