from shutil import rmtree

from ..utils import results_path


def clear_subs() -> None:
    """
    Deletes preprocessings, results and figures.
    """
    for path in [results_path]:
        if path.exists():
            rmtree(path)
