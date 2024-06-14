from pathlib import Path
from shutil import rmtree

from ..utils import descriptions_base_path, results_path


def clear_subs() -> None:
    """
    Deletes preprocessings, results and figures.
    """
    path: Path
    for path in [results_path, descriptions_base_path]:
        if path.exists():
            rmtree(path)
