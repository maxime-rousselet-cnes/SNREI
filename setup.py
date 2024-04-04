from setuptools import setup

setup(
    name="snrei",
    version="0.1",
    description="A Python package for Spherically symmetric non-rotating and isotropic Earth",
    author="Maxime Rousselet - CNES",
    author_email="maxime.rousselet@cnes.fr",
    packages=["snrei"],
    install_requires=[
        "numpy",
        "matplotlib",
        "netCDF4",
        "scipy",
        "shutil",
        "itertools",
        "json",
        "pathlib",
        "pydantic",
        "multiprocessing",
    ],
)
