import torch
import numpy as np

from utils.LieGroup_torch_v2 import *
from utils.torch_utils import *
from robot.openchains_torch import *

## For Mass Computes
import pybullet as p
import pybullet_data

class Franka:
    def __init__(self, device='cpu', _robot_body_id=None):
        self.A_screw = torch.tensor([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=torch.float32).to(device) 
        
        
        self.M = torch.zeros(7, 4, 4).to(device) 
        self.M[0] = torch.tensor([[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0.333], 
                [0, 0, 0, 1]]).to(device) 

        self.M[1] = torch.tensor([[1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0,-1, 0, 0], 
                [0, 0, 0, 1.0]]).to(device) 

        self.M[2] = torch.tensor([[1, 0, 0, 0], 
                [0, 0, -1, -0.316], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 

        self.M[3] = torch.tensor([[1, 0, 0, 0.0825], 
                [0, 0,-1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 

        self.M[4] = torch.tensor([[1, 0, 0, -0.0825], 
                [0, 0, 1, 0.384], 
                [0,-1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 
            
        self.M[5] = torch.tensor([[1, 0, 0, 0], 
                [0, 0,-1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1.0]]).to(device) 

        self.M[6] = torch.tensor([[1, 0, 0, 0.088], 
                [0, 0, -1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 

        self.initialLinkFrames_from_base = torch.zeros(7, 4, 4).to(device) 
        self.initialLinkFrames_from_base[0] = self.M[0]
        for i in range(1, 7):
            self.initialLinkFrames_from_base[i] = self.initialLinkFrames_from_base[i-1]@self.M[i]
        
        self.LLtoEE = torch.tensor(
                [[0.7071, 0.7071, 0, 0], 
                [-0.7071, 0.7071, 0, 0], 
                [0, 0, 1, 0.107], 
                [0, 0, 0, 1]]).to(device)
        
        self.EEtoLeftFinger = torch.tensor(
                [[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0.0584], 
                [0, 0, 0, 1]]).to(device)
        
        self.EEtoRightFinger = torch.tensor(
                [[-1, 0, 0, 0], 
                [0, -1, 0, 0], 
                [0, 0, 1, 0.0584], 
                [0, 0, 0, 1]]).to(device)
        
        self.initialEEFrame = self.initialLinkFrames_from_base[-1]@self.LLtoEE
        
        self.S_screw = compute_S_screw(
                self.A_screw, 
                self.initialLinkFrames_from_base
        )
        
        self.inertias = torch.zeros(7, 6, 6).to(device)
        if _robot_body_id is not None:
            for i in range(7):
                results = p.getDynamicsInfo(_robot_body_id, i)
                m = results[0] 
                (ixx, iyy, izz) = results[2]
                self.inertias[i] = torch.tensor(
                        [
                                [ixx, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, iyy, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, izz, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, m, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, m, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, m]
                        ]
                ).to(device)
                
        #############################################
        #################### Limits #################
        # https://frankaemika.github.io/docs/control_parameters.html
        self.JointPos_Limits = [
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], 
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.752, 2.8973]]
        self.JointVel_Limits = [
                2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        self.JointAcc_Limits = [
                15, 7.5, 10, 12.5, 15, 20, 20
        ]
        self.JointJer_Limits = [
                7500, 3750, 5000, 6250, 7500, 10000, 10000
        ]
        self.JointTor_Limits = [
                87, 87, 87, 87, 12, 12, 12
        ]
        
        self.CarVelocity_Limits = [
                2*2.5, 2*1.7 
        ]
        #############################################
        #############################################
        
        #############################################
        ############ Collision capsules #############
        self.link_capsules = [
            [-0.05713602,  0.00197699,  0.01697141, 0.00026363, -0.00135329,  0.05965031, 0.10656500607728958],
            [-0.00020432,  0.00631868, -0.14863949, 0.00130182, -0.05217961, -0.00339109, 0.07845212519168854],
            [-1.1685867e-03, -1.4363851e-01, 7.7432787e-07, 0.00129307, 0.00214446, 0.05342388, 0.07673688977956772],
            [0.00113084, 0.00143992, -0.07623108, 0.08554848, 0.04127572, 0.00466059, 0.07257714867591858],
            [-0.08218852, 0.08021478, 0.00232926, -0.00809084, 0.00193282, 0.04148004, 0.07551463693380356],
            [0.004206, 0.00960619, -0.24048167, 9.8637734e-05, 6.2054291e-02, -3.4491196e-03, 0.07147178053855896],
            [0.02307169, -0.00108738, -0.00042872, 0.07747807, 0.00616371, 0.00199851, 0.0793522447347641],
            [0.03198181, 0.03163229, 0.06881348, -0.00696545, -0.00611855, 0.07171073, 0.05475013703107834],
        ]
        self.gripper_capsules = [
                [0.00520471, -0.04667679, 0.03506533, -0.00544212, 0.04179848, 0.04411257, 0.07438668608665466]
        ]
        #############################################
        #############################################