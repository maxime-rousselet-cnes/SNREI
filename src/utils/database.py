from json import JSONEncoder, dump, load
from pathlib import Path
from typing import Any, Optional

from numpy import arange, concatenate, ndarray
from pydantic import BaseModel


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
    if len(name.split("/")) > 1:
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
    with open(path.joinpath(name + ".json"), "r") as file:
        loaded_content = load(fp=file)
        return loaded_content if not base_model_type else base_model_type(**loaded_content)


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
