import re
import pandas as pd
import numpy as np
from datasets import Dataset, Features, Sequence, Value
import os
import json
import glob
import shutil

def csv_to_arrow(csv_files, output_dir):
    datasets = []
    total_examples = 0
    shard_lengths = []
    all_arrow_filenames = []

    features = Features({
        "id": Value(dtype="string"),
        "target": Sequence(feature=Value(dtype="float32"))
    })

    for csv_file in csv_files:

        df = pd.read_csv(csv_file,header=None)
        if df.isna().any().any():
            pass
        all_series = []
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        for column in df.columns:
            if df[column].isna().any():
                pass
            series = df[column].dropna().astype(np.float32).tolist()
            all_series.append({
                "id": csv_basename,
                "target": series
            })

        dataset = Dataset.from_list(all_series, features=features)
        datasets.append(dataset)
        total_examples += len(dataset)
        shard_lengths.append(len(dataset))

        arrow_filename = f"{csv_basename}.arrow"
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        dataset.save_to_disk(temp_dir,num_shards=1)
        temp_arrow_files = glob.glob(os.path.join(temp_dir, "*.arrow"))
        if len(temp_arrow_files) == 1:
            os.rename(temp_arrow_files[0], os.path.join(output_dir, arrow_filename))
        else:
            def get_shard_index(filename):
                match = re.search(r'data-(\d+)-of-\d+', filename)
                if match:
                    return int(match.group(1))
                return float('inf')

            temp_arrow_files = sorted(temp_arrow_files, key=get_shard_index)

            for i, arrow_file in enumerate(temp_arrow_files, 1):
                new_filename = f"{csv_basename}{i}.arrow"
                new_path = os.path.join(output_dir, new_filename)
                if os.path.exists(new_path):
                    raise FileExistsError(f" {new_path} ")
                os.rename(arrow_file, new_path)
                all_arrow_filenames.append(new_filename)


        shutil.rmtree(temp_dir)
        all_arrow_filenames.append(arrow_filename)
        print(csv_file)
    state = {
        "_data_files": f"{self.name}.arrow",
        "_fingerprint": datasets[0]._fingerprint,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": "train"
    }
    with open(os.path.join(output_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)

    total_size = sum(sum(dataset.data.nbytes for table in dataset.data.table) for dataset in datasets)
    dataset_info = {
        "builder_name": "generator",
        "citation": "",
        "config_name": "default",
        "dataset_name": "CustomCSV",
        "dataset_size": total_size,
        "description": "CSV",
        "download_checksums": {},
        "download_size": 0,
        "features": {
            "id": {
                "dtype": "string",
                "_type": "Value"
            },
            "target": {
                "feature": {
                    "dtype": "float32",
                    "_type": "Value"
                },
                "_type": "Sequence"
            }
        },
        "homepage": "",
        "license": "",
        "size_in_bytes": total_size,
        "splits": {
            "train": {
                "name": "train",
                "num_bytes": total_size,
                "num_examples": total_examples,
                "shard_lengths": shard_lengths,
                "dataset_name": "generator"
            }
        },
        "version": {
            "version_str": "0.0.0",
            "major": 0,
            "minor": 0,
            "patch": 0
        }
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)


csv_files = glob.glob(os.path.join(input_dir, "**", "*_te.csv"), recursive=True)
csv_to_arrow(csv_files, output_dir)
