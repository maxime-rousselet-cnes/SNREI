# Clears dataset folders:
#   > data
#       > descriptions
#       > results

from shutil import rmtree

from utils import descriptions_path, figures_path, results_path


def clear_products() -> None:
    """
    Deletes preprocessings and results.
    """
    for path in [descriptions_path, figures_path, results_path]:
        if path.exists():
            rmtree(path)


if __name__ == "__main__":
    clear_products()
