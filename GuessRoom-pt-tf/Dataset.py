# NOTE assume that agentA is in the middle of each room
import os
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Param import *
import numpy as np

# TODO: check read order
class EnvDataset(Dataset):
    def __init__(self, env_dir):
        self.env_dir = env_dir
        self.env_files = os.listdir(self.env_dir)
        self.env_files = [f for f in self.env_files if not f.startswith(".")]  # remove .DS_Store
    def __getitem__(self, item):
        with open("{}/obs{}.txt".format(self.env_dir, item), 'rb') as f:
            data=np.load(file=f)
            return data
            
    def __len__(self):
        return len(self.env_files) 
