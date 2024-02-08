import argparse

from utils import Love_numbers_from_models_to_result

parser = argparse.ArgumentParser()
parser.add_argument("real_description_id", type=str, help="Optional wanted ID for the real description")
parser.add_argument("run_id", type=str, help="Optional wanted ID for the run")
parser.add_argument(
    "load_description", type=bool, action="store_true", help="Option to tell if the description should be loaded"
)

args = parser.parse_args()


if __name__ == "__main__":
    print(
        Love_numbers_from_models_to_result(
            real_description_id=args.real_description_id, run_id=args.run_id, load_description=args.load_description
        )
    )
