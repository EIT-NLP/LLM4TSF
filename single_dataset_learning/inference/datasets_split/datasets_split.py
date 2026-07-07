import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class Datasets_split(Dataset):
    def __init__(self, size, dataset):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.dataset = dataset
        self.windows = [len(seq) - self.seq_len-self.pred_len + 1 for seq in dataset['target']]
        self.total_len = sum(self.windows)
        self.single_len = self.windows[0]
    def __getitem__(self, index):
        col_id = (int)(index // self.single_len)
        col_index = (int)(index % self.single_len)
        s_begin = col_index
        timeseries = self.dataset[col_id]['target']
        timeseries = np.array(timeseries)
        s_end = s_begin + self.seq_len
        r_begin = s_begin
        r_end = s_begin +self.pred_len+self.seq_len
        seq_x = timeseries[s_begin:s_end]
        seq_y = timeseries[r_begin:r_end]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        return seq_x, seq_y

    def __len__(self):
        return self.total_len








