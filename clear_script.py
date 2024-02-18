# Clears dataset folders:
#   > data
#       > descriptions
#       > results

from shutil import rmtree

from utils import descriptions_path, results_path


def clear_products() -> None:
    """
    Deletes preprocessings and results.
    """
    if descriptions_path.exists():
        rmtree(descriptions_path)
    if results_path.exists():
        rmtree(results_path)


if __name__ == "__main__":
    clear_products()
