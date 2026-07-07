import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
start_time = time.time()
seq_len = 96
def is_missing(df_percol):
    has_missing = df_percol.isna().any()
    missing_count = df_percol.isna().sum()
    print(f"Column {col}:")
    print(f"  Has missing values: {has_missing}")
    print(f"  Number of missing values: {missing_count}")
scaler = StandardScaler()
dataset_name = "PEMS04"
base_read_path = "dataset/PEMS"
path = f"{base_read_path}/{dataset_name}.npz"

base_output_path = "dataset_split"
output_dir1 = f"{base_output_path}/{dataset_name}_tr.csv"
output_dir2 = f"{base_output_path}/{dataset_name}_va.csv"
output_dir3 = f"{base_output_path}/{dataset_name}_te.csv"

npz = np.load(path)
data0 = npz['data']
data = data0.reshape(data0.shape[0], -1)
df = pd.DataFrame(data)
train_data_list = []
valid_data_list = []
test_data_list = []
results = []
train_df = pd.DataFrame()
valid_df = pd.DataFrame()
test_df = pd.DataFrame()
cnt = 0
for col in df.columns:
    df_percol = df[col]
    # while len(df_percol) > 0 and pd.isna(df_percol.iloc[-1]):
    #     df_percol = df_percol.iloc[:-1]
    # is_missing(df_percol)
    if df_percol.isna().any():
        df_percol1 = df_percol.fillna(method="bfill")
        df_percol2 = df_percol1.dropna()
        df_percol = df_percol2
    if dataset_name == "ETTh1" or dataset_name == "ETTh2":
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset_name == "ETTm1" or dataset_name == "ETTm2":
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        df_percol_length = len(df_percol)
        num_train = int(df_percol_length * 0.7)
        num_test = int(df_percol_length * 0.2)
        num_vali = df_percol_length - num_train - num_test
        border1s = [0, num_train - seq_len, df_percol_length - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, df_percol_length]
    if cnt == 0:
        print(dataset_name)
        cnt += 1
    train_data = df_percol[border1s[0]:border2s[0]].reset_index(drop=True)
    valid_data = df_percol[border1s[1]:border2s[1]].reset_index(drop=True)
    test_data = df_percol[border1s[2]:border2s[2]].reset_index(drop=True)

    train_data_list.append(pd.Series(train_data))
    valid_data_list.append(pd.Series(valid_data))
    test_data_list.append(pd.Series(test_data))

train_data_combined = pd.concat(train_data_list,axis=1)
valid_data_combined = pd.concat(valid_data_list,axis=1)
test_data_combined = pd.concat(test_data_list,axis=1)
train_data_combined.to_csv(output_dir1,index=False,header=False)
valid_data_combined.to_csv(output_dir2,index=False,header=False)
test_data_combined.to_csv(output_dir3,index=False,header=False)