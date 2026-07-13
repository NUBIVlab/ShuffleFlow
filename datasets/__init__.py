import re
from typing import List

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, name_data: str, name_operator: str, idx_list: List[int]=None, **kwargs):
        self.name_data = name_data
        self.name_operator = name_operator
        
        if idx_list is None:
            self.idx_map = lambda x: x
            self.idx_list = list[int](range(100))
        else:
            idx_list = self.parse_int_list(idx_list)
            self.idx_map = lambda x: idx_list[x]
            self.idx_list = idx_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        pass
    
    @staticmethod
    def parse_int_list(s):
        if isinstance(s, list): return s
        if isinstance(s, int): return [s]
        ranges = []
        range_re = re.compile(r'^(\d+)-(\d+)$')
        for p in s.split(','):
            m = range_re.match(p)
            if m:
                ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
            else:
                ranges.append(int(p))
        return ranges
