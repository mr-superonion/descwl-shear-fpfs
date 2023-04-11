import os
from setuptools import setup, find_packages

# version of the package
__version__ = "0.1"


scripts = [
]

setup(
    name="deswl-shear-fpfs",
    version=__version__,
    description="Auto-diff Estimator of Lensing Perturbations",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "schwimmbad",
        "jax",
        "flax",
        "fpfs",
        "fitsio",
        "pre-commit",
        "matplotlib"
    ],
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/mr-superonion/descwl-shear-fpfs",
)
