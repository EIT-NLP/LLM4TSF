import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import Dataset, Features, Sequence, Value
import os
import json


class test_sets:
    def __init__(self, name, read_path, output_path):
        self.df = pd.read_csv(read_path,header=None)
        self.name = name
        self.output_path = output_path

    def plot_columns(self):
        for i, col in enumerate(self.df.columns, start=1):
            plt.figure(figsize=(10, 6))
            plt.plot(self.df_normalized[col], linestyle='-', label=col)
            plt.grid(True)
            save_path = os.path.join(self.output_path, f"{self.name}_{i}.jpg")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

    def csv_to_arrow(self):
        output_path = self.output_path
        total_examples = 0
        shard_lengths = []

        features = Features({
            "id": Value(dtype="string"),
            "target": Sequence(feature=Value(dtype="float32"))
        })
        df = self.df_normalized
        all_series = []
        for column in df.columns:
            series = df[column].dropna().astype(np.float32).tolist()
            all_series.append({
                "id": self.name,
                "target": series
            })

        dataset = Dataset.from_list(all_series, features=features)
        arrow_filename = f"{self.name}"
        dataset.save_to_disk(arrow_filename,num_shards=1)
        state = {
            "_data_files": f"{self.name}.arrow",
            "_fingerprint": dataset._fingerprint,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": "train"
        }
        with open(os.path.join(output_path, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        dataset_info = {
            "builder_name": "generator",
            "citation": "",
            "config_name": "default",
            "dataset_name": "CustomCSV",
            "description": "from",
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
            "splits": {
                "train": {
                    "name": "train",
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
        with open(os.path.join(output_path, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)

    def normalize(self):
        self.df_normalized = self.df.copy()
        stats_data = {
            'column': [],
            'mean': [],
            'std': []
        }
        for col in self.df.columns:
            series = self.df[col].dropna().astype(np.float32).tolist()
            if len(series) < 336 +96:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                if std_val == 0:
                    normalized_col = self.df[col]
                else:
                    normalized_col = (self.df[col] - mean_val) / std_val

                self.df_normalized[col] = normalized_col

                stats_data['column'].append(col)
                stats_data['mean'].append(mean_val)
                stats_data['std'].append(std_val)
            else:
                print(f"'{col}' ")
        stats_filename = f"{self.name}_stats.csv"
        stats_filepath = os.path.join(self.output_path, stats_filename)
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_filepath, index=False, encoding='utf-8')

if __name__ == "__main__":
    name = "Smart"
    read_path = f"{name}.csv"
    output_path = f"{name}"
    ts = test_sets(name, read_path, output_path)
    ts.normalize()
    ts.csv_to_arrow()
    print(1)