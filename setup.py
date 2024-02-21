from setuptools import setup

setup(
    name="snrei",
    version="0.1",
    description="A Python package for Spherically symetric non-rotating and isotropic Earth",
    author="Maxime Rousselet - CNES",
    author_email="maxime.rousselet@cnes.fr",
    packages=["snrei"],
    install_requires=[
        "numpy",
        "matplotlib",
        "netCDF4",
        "scipy",
        "shutil",
        "argparse",
        "itertools",
        "json",
        "pathlib",
        "uuid",
        "pydantic",
        "multiprocessing",
    ],
)
