import numpy as np
import torch

import sys
sys.path.append('../')
from utils import LieGroup_numpy, LieGroup_torch

class Franka:
    def __init__(self, franka_xml_path, surface_point_cloud=None):
        import xml.etree.ElementTree as ET
        tree = ET.parse(franka_xml_path)
        root = tree.getroot()
        panda_link1 = root[5][0][2]
        panda_link2 = root[5][0][2][4]
        panda_link3 = root[5][0][2][4][4]
        panda_link4 = root[5][0][2][4][4][4]
        panda_link5 = root[5][0][2][4][4][4][4]
        panda_link6 = root[5][0][2][4][4][4][4][4]
        panda_link7 = root[5][0][2][4][4][4][4][4][4]
        ee_site     = root[5][0][2][4][4][4][4][4][4][4]
        panda_hand  = root[5][0][2][4][4][4][4][4][4][5]

        self.Phi = np.array([
            [float(panda_link[0].attrib['mass'])] + \
            [float(x) for x in panda_link[0].attrib['pos'].split(' ')] + \
            [float(x) for x in panda_link[0].attrib['fullinertia'].split(' ')] for panda_link in [
                panda_link1, panda_link2, panda_link3, panda_link4, panda_link5, panda_link6, panda_link7
            ]])
        self.A_screw = np.array([
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]) 
        
        def compute_initial_SE3(attrib):
            p = np.array([float(x) for x in attrib['pos'].split(' ')])
            try:
                R = LieGroup_torch.quaternions_to_rotation_matrices_torch(
                    torch.tensor([[float(x) for x in attrib['quat'].split(' ')]])
                ).squeeze(0).numpy()
            except:
                R = np.eye(3)
            T = np.vstack(
                [np.hstack([R, p.reshape(3,1)]),
                np.array([0, 0, 0, 1])]
            )
            return T

        self.initialLinkFrames = np.array(
                [
                    compute_initial_SE3(link.attrib) for link in [
                        panda_link1, panda_link2, panda_link3, panda_link4, panda_link5, panda_link6, panda_link7
                    ]
                ]
            )
        
        
        self.initialEEFrame = self.initialLinkFrames[0]@self.initialLinkFrames[1]@self.initialLinkFrames[2]@self.initialLinkFrames[3]@\
                            self.initialLinkFrames[4]@self.initialLinkFrames[5]@self.initialLinkFrames[6]@compute_initial_SE3(ee_site.attrib)
        
        
        initialLinkFrames_from_base = []
        temp = np.eye(4)
        for frame in self.initialLinkFrames:
            temp = temp@frame
            initialLinkFrames_from_base.append(
                temp.reshape(1,4,4)
            )
        self.initialLinkFrames_from_base = np.concatenate(initialLinkFrames_from_base, axis=0)
        
        S_screw = []
        for M, A in zip(self.initialLinkFrames_from_base, self.A_screw):
            S_temp = LieGroup_numpy.Adjoint_SE3(M)@A
            S_screw.append(S_temp.reshape(1,6))
        self.S_screw = np.concatenate(S_screw, axis=0)
        self.B_screw = (LieGroup_numpy.Adjoint_SE3(
            LieGroup_numpy.inv_SE3(self.initialEEFrame)
        )@self.S_screw.transpose()).transpose()
        
        self.joint_limit = np.array([
                            [float(x) for x in panda_link1[1].attrib['range'].split(' ')], \
                            [float(x) for x in panda_link2[1].attrib['range'].split(' ')], \
                            [float(x) for x in panda_link3[1].attrib['range'].split(' ')], \
                            [float(x) for x in panda_link4[1].attrib['range'].split(' ')], \
                            [float(x) for x in panda_link5[1].attrib['range'].split(' ')], \
                            [float(x) for x in panda_link6[1].attrib['range'].split(' ')], \
                            [float(x) for x in panda_link7[1].attrib['range'].split(' ')]    
                        ])
        
        if surface_point_cloud is not None:
            self.surface_point_cloud = surface_point_cloud