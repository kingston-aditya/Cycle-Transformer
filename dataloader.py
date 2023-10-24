import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset

# only for test, not required actually

# x = torch.rand((500, 1, 512))
# y = torch.rand((500, 1, 512))

# torch.save(x, 'train_image.pt')
# torch.save(y, 'train_text.pt')

# real work
class data_make(Dataset):
    
    def __init__(self, root_img, root_txt, data_length):
        self.root_img = root_img
        self.root_txt = root_txt
        self.length_dataset = data_length
        
    def __len__(self):
        return self.length_dataset
        
    def __getitem__(self, index):
        
        x = torch.load(self.root_img)
        y = torch.load(self.root_txt)
        
        x_1 = x[index]
        y_1 = y[index]
        
        return(x_1,y_1)