import torch
import numpy as np
import os, sys
from omegaconf import OmegaConf
from matplotlib.patches import Circle

class Robot(torch.utils.data.Dataset):
    def __init__(self, 
        root,
        split='training',
        **kwargs):

        super(Robot, self).__init__()

        data = []
        targets = []
        for file_ in os.listdir(root):
            if file_.endswith('.npy'):
                traj_data = np.load(os.path.join(root, file_))
                data.append(torch.tensor(traj_data, dtype=torch.float32).unsqueeze(0))
                
                if file_.startswith('lower'):
                    targets.append(torch.tensor([0]).view(1, 1))
                elif file_.startswith('upper'):
                    targets.append(torch.tensor([1]).view(1, 1))
                elif file_.startswith('none'):
                    targets.append(torch.tensor([
                        int(file_.split('.')[0].split('_')[-1])
                        ]).view(1, 1))
                    
        data = torch.cat(data, dim=0)
        targets = torch.cat(targets, dim=0)
        self.data = data
        self.targets = targets
        
        # zero_idx = (targets==0).view(-1)
        # if split == 'training':
        #     self.data_ = torch.cat([data[zero_idx][:4], data[~zero_idx][:4]], dim=0)
        #     self.targets_ = torch.cat([targets[zero_idx][:4], targets[~zero_idx][:4]], dim=0)
        # elif split == 'validation':
        #     self.data_ = torch.cat([data[zero_idx][4:], data[~zero_idx][4:]], dim=0)
        #     self.targets_ = torch.cat([targets[zero_idx][4:], targets[~zero_idx][4:]], dim=0)
            
        print(f"Robot split {split} | {self.data.size()}")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y