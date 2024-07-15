from os import path, remove, rmdir, walk
from os.path import exists
from pathlib import Path

from ..utils import descriptions_base_path, results_path


def clear_subs() -> None:
    """
    Deletes preprocessings, results and figures.
    """
    path: Path
    for path in [results_path, descriptions_base_path]:
        if path.exists():
            remove_folder(path)


import os


def remove_folder(folder_path):
    """
    Remove a folder and all its contents.

    Parameters:
    folder_path (str): The path to the folder to be removed.

    Returns:
    None
    """
    if exists(folder_path):
        for root, dirs, files in walk(folder_path, topdown=False):
            for name in files:
                remove(path.join(root, name))
            for name in dirs:
                rmdir(path.join(root, name))
        rmdir(folder_path)
