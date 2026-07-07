import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# Due to varying sequence lengths across different datasets, binary search is employed to locate the corresponding sequence in the shortest possible time.
class Datasets_split(Dataset):
    def __init__(self, size, dataset):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.dataset = dataset
        self.windows = [len(seq) - self.seq_len-self.pred_len + 1 for seq in dataset['target']]
        self.total_len = sum(self.windows)
        self.single_len = self.windows[0]

    def __getitem__(self, index):
        index += 1
        window_index, inter_window_id = self.find_index(index)
        col_id = window_index
        col_index = inter_window_id
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

    def idx(self,target, window_sums):
        idx = bisect.bisect_left(window_sums, target)
        if idx == 0:
            return None
        return idx - 1

    def find_index(self,index):
        arr_list = self.windows
        window_sums = np.cumsum(arr_list).tolist()
        window_leftid = [0] + window_sums[:-1]
        window_index = self.idx(index, window_leftid)
        inter_window_id = index - window_leftid[window_index]-1
        return window_index, inter_window_id







