from setuptools import find_packages, setup

setup(
    name="SNREI",
    version="7",
    description="A Python package for Spherically symmetric non-rotating and isotropic Earth",
    author="Maxime Rousselet - CNES",
    author_email="maxime.rousselet@cnes.fr",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "netCDF4",
        "scipy",
        "more-itertools==8.10.0",
        "pathlib",
        "pydantic",
        "multiprocess",
        "cmocean",
    ],
)
