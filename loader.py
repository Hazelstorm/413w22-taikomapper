import torch
from torch.utils.data import Dataset, DataLoader
import os
from helper import *

class MapDataset(Dataset):
    def __init__(self, path, start, end):
        self.path = path
        self.dir = os.listdir(path)
        self.start = round(len(self.dir) * start)
        self.end = round(len(self.dir) * end)
        
    def __len__(self):
        return self.end - self.start
    
    def __getitem__(self, idx):
        song_path = os.path.join(self.path, self.dir[idx + self.start])
        audio_data, timing_data, notes_data = get_npy_data(song_path)
        audio_data, notes_data = torch.Tensor(audio_data), torch.Tensor(notes_data)
        mask = torch.zeros(audio_data.shape[1], audio_data.shape[1])
        
        return audio_data, notes_data
    
def get_dataloader(path, start, end, batch_size=1, shuffle=False):
    dataset = MapDataset(path, start, end)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
    