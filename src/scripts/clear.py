from shutil import rmtree

from ..utils import descriptions_base_path, figures_path, results_path


def clear_subs() -> None:
    """
    Deletes preprocessings, results and figures.
    """
    for path in [descriptions_base_path, figures_path, results_path]:
        if path.exists():
            rmtree(path)
