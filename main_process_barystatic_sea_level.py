import json
import os

import pandas as pd


def process_files(main_folder):
    data = {}
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    json_data = json.load(f)
                    parent_folders = root.split(os.sep)[-4:-1]  # Get the names of the 3 parent folders
                    column_label = root.split(os.sep)[-5]  # Get the name of the 5th parent folder
                    row_label = "_".join(parent_folders)  # Combine the names of the parent folders
                    if row_label not in data:
                        data[row_label] = {}
                    data[row_label][column_label] = json_data["b"] / json_data["a"]

    # Create DataFrame from the collected data
    df = pd.DataFrame(data).T

    return df


# Example usage:
main_folder = "/path/to/main/folder"
df = process_files(main_folder)
print(df)
