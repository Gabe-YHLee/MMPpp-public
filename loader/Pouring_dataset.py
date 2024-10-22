import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os, sys
import pickle

import math

from utils.LieGroup_torch import *
from utils.utils import SE3smoothing

class Pouring(Dataset):
    def __init__(
            self, 
            split='training', 
            root='datasets/bigbottle_pkl_subdataset/pouring_data',
            **kwargs
        ):
        
        self.traj_data = [] # traj data (len x 4 x 4)
        self.labels = [] # binary lables (e.g., wine style or not, a lot of water or small, etc)
        
        for file_ in os.listdir(root):
            with open(os.path.join(root, file_), "rb") as f:
                data = pickle.load(f)
                traj = data['traj']
                traj = traj@np.array(
                        [[
                            [1., 0., 0., data['offset'][0]], 
                            [0., 1., 0., data['offset'][1]], 
                            [0., 0., 1., data['offset'][2]], 
                            [0., 0., 0., 1.]]])
                
                
                self.traj_data.append(torch.tensor(traj, dtype=torch.float32).unsqueeze(0))
                self.labels.append(
                        torch.tensor(data['label']).unsqueeze(0)) # Temporary
                    
        self.traj_data = torch.cat(self.traj_data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)[:, -1]
        
    def __getitem__(self, idx):
        traj = self.traj_data[idx] # (Len, 4, 4)
        labels = self.labels[idx] 
        return traj, labels

    def __len__(self) -> int:
        return len(self.traj_data)
    
class HandMadePouring(Dataset):
    def __init__(
            self, 
            split='training', 
            **kwargs
        ):
        self.traj_data = [] # traj data (len x 4 x 4)
        self.labels = [] # binary lables (e.g., wine style or not, a lot of water or small, etc)
        
        T_init = torch.tensor([
            [ 1,  0,  0,  0.6],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0.3],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

        T_final = torch.tensor([
                [ math.cos(2.3*3.141592/4),  0,  -math.sin(2.3*3.141592/4),  0.3],
                [ 0,  1,  0,  0],
                [ math.sin(2.3*3.141592/4),  0,  math.cos(2.3*3.141592/4),  0.23],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

        t = torch.linspace(0, 1, 480).view(-1, 1)

        p_init = T_init[:3, 3]
        R_init = T_init[:3, :3]
        p_final = T_final[:3, 3]
        R_final = T_final[:3, :3]
        
        for width, label in zip([1.0, 0.5, -0.5, -1.0], [0, 0, 1, 1]):
            for k in [-0.8, 0, 0.8]:
                p_traj = (1-t)*p_init.view(1, 3) + t*p_final.view(1,3) + 4*t**2*(1-t)**2*torch.tensor([[0, width, k]], dtype=torch.float32)
                R_traj = R_init.view(1, 3, 3)@exp_so3(log_SO3(R_init.permute(1, 0).view(1, 3, 3)@R_final.view(1, 3, 3)) * t.view(-1, 1, 1)**2)
                R_traj = R_traj@exp_so3(-4*width*t*(1-t)*torch.tensor([[0, 0, 1]], dtype=torch.float32))
                
                p_traj = p_traj + torch.randn_like(p_traj) * 0.001
                R_traj = exp_so3(log_SO3(R_traj) + skew(torch.randn_like(p_traj)) * 0.001)
                
                T_traj = torch.cat([R_traj, p_traj.view(-1, 3, 1)], dim=-1)
                T_traj = torch.cat(
                    [
                        T_traj,
                        torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 1, 4).repeat(len(T_traj), 1, 1)
                    ], dim=1
                )
        
                self.traj_data.append(T_traj.view(1, -1, 4, 4))
                self.labels.append(torch.tensor([label], dtype=torch.int))
        
        self.traj_data = SE3smoothing(torch.cat(self.traj_data, dim=0))
        self.labels = torch.cat(self.labels, dim=0)
        
    def __getitem__(self, idx):
        traj = self.traj_data[idx] # (Len, 4, 4)
        labels = self.labels[idx] 
        return traj, labels

    def __len__(self) -> int:
        return len(self.traj_data)