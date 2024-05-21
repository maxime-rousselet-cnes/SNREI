from contextlib import suppress
from json import JSONEncoder, dump, load
from os import remove
from os import symlink as base_symlink
from pathlib import Path
from typing import Any, Optional

from numpy import arange, complex128, concatenate, fromfile, ndarray
from pydantic import BaseModel


def symlink(src: Path, dst: Path):
    """
    Creates a Symlink, eventually remove a previously existing one.
    """
    with suppress(FileNotFoundError):
        remove(dst)
    base_symlink(src=src, dst=dst)


def symlinkfolder(src: Path, dst: Path) -> None:
    """
    Recursively creates symlinks from folder to folder.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        new_child = dst.joinpath(child.name)
        if child.is_file():
            symlink(src=child, dst=new_child)
        else:
            symlinkfolder(src=child, dst=new_child)


class JSONSerialize(JSONEncoder):
    """
    Handmade JSON encoder that correctly encodes special structures.
    """

    def default(self, obj: Any):
        if type(obj) is ndarray:
            return obj.tolist()
        elif isinstance(obj, BaseModel):
            return obj.__dict__
        else:
            JSONEncoder().default(obj)


def save_base_model(obj: Any, name: str, path: Path):
    """
    Saves a JSON serializable type.
    """
    # May create the directory.
    while len(name.split("/")) > 1:
        path = path.joinpath(name.split("/")[0])
        name = "".join(name.split("/")[1:])
    path.mkdir(exist_ok=True, parents=True)
    # Saves the object.
    with open(path.joinpath(name + ".json"), "w") as file:
        dump(obj, fp=file, cls=JSONSerialize)


def load_base_model(
    name: str,
    path: Path,
    base_model_type: Optional[Any] = None,
) -> Any:
    """
    Loads a JSON serializable type.
    """
    filepath = path.joinpath(name + ("" if ".json" in name else ".json"))
    while filepath.is_symlink():
        filepath = filepath.resolve()
    with open(filepath, "r") as file:
        loaded_content = load(fp=file)
        return loaded_content if not base_model_type else base_model_type(**loaded_content)


def save_complex_array_to_binary(array: ndarray, name: str, path: Path) -> None:
    """
    Saves a complex NumPy array to a binary file.
    """
    path.mkdir(parents=True, exist_ok=True)
    filename = path.joinpath(name)
    with open(filename, "wb") as f:
        array.tofile(f)


def load_complex_array_from_binary(name: str, path: Path) -> ndarray[complex128]:
    """
    Loads a complex NumPy array from a binary file.
    """
    # Load the array from binary file
    filename = path.joinpath(name)
    return fromfile(filename, dtype=complex128)


def generate_degrees_list(
    degree_thresholds: list[int],
    degree_steps: list[int],
) -> list[int]:
    """
    Generates the list of degrees for which to compute Love numbers, given a list of thresholds and a list of steps.
    """
    return concatenate(
        [
            arange(degree_thresholds[i], degree_thresholds[i + 1], degree_step, dtype=int)
            for i, degree_step in enumerate(degree_steps)
        ],
        dtype=int,
    ).tolist()


def get_run_folder_name(anelasticity_description_id: str, run_id: str) -> str:
    """
    Builds a folder id with anelasticity description id and run id.
    """
    return "/".join((anelasticity_description_id, "runs", run_id))
