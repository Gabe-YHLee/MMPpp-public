import open3d as o3d
import os
import numpy as np
import robot.kinematics_for_franka as lie_alg
from robot.kinematics_for_franka import forward_kinematics, inverse_kinematics
from robot.franka_sub import Franka as FSub
import sys
sys.path.append('..')
import robot.Kinematics_numpy as Kinematics_numpy
from vis_utils.open3d_utils import (
    get_mesh_bottle, 
)
from robot.robot_utils import(
    traj_robotbase_EE_to_bottle_numpy_single,
)
from copy import deepcopy
import torch

class Franka:
    def __init__(self, azure=True, robot_color = [0.9, 0.9, 0.9], 
            azure_color = [0.4, 0.4, 0.4], bracket_color = [0.1, 0.1, 0.1],
            root='', add_bottle=False, bottle_grasp_height = 0.12,
            pcd=False):
        self.franka_sub = FSub(os.path.join(root, 'robot_vis/franka_panda_robot_only_cat.xml'))
        self.robot_color = robot_color
        self.azure_color = azure_color
        self.bracket_color = bracket_color
        self.azure = azure
        self.pcd = pcd
        if add_bottle == True:
            self.mesh_bottle = get_mesh_bottle(
                                root=os.path.join(root, '3dmodels/bottles'), 
                                bottle_idx=1)
            self.mesh_bottle.paint_uniform_color([1, 0, 0])
            grasp_height = bottle_grasp_height
            T_grasp = np.eye(4)
            T_grasp[2, 3] = -grasp_height
            T_rot_m90_x = np.eye(4)
            T_rot_m90_x[2, 3] = -grasp_height
            self.mesh_bottle.transform(T_grasp)
            self.franka_EE_frame = self.franka_sub.initialEEFrame
            self.bottle_SE3 = traj_robotbase_EE_to_bottle_numpy_single(self.franka_EE_frame)
            self.mesh_bottle.transform(self.bottle_SE3)
        else:
            self.mesh_bottle = None

        if azure:
            n_link = 10
        else:
            n_link = 8
        self.n_link = n_link
        self.link = [None] * n_link

        for link_num in range(8):
            self.link[link_num] = o3d.io.read_triangle_mesh(os.path.join(root, f"robot_vis/mesh/link{link_num}.ply"))
            self.link[link_num].compute_vertex_normals()
            self.link[link_num].paint_uniform_color(self.robot_color)
        
        self.M = [None] * n_link

        self.M[0] = np.identity(4)
        self.M[1] = lie_alg.define_SE3(
            np.identity(3), [0, 0, 0.333])
        self.M[2] = lie_alg.define_SE3(
            lie_alg.exp_so3(np.array([1, 0, 0]) * (- np.pi / 2)), [0, 0, 0.333])
        self.M[3] = lie_alg.define_SE3(
            np.identity(3), [0, 0, 0.333 + 0.316])
        self.M[4] = lie_alg.define_SE3(
            lie_alg.exp_so3(np.array([1, 0, 0]) * (np.pi / 2)), [0.0825, 0, 0.333 + 0.316])
        self.M[5] = lie_alg.define_SE3(
            np.identity(3), [0, 0, 0.333 + 0.316 + 0.384])
        self.M[6] = lie_alg.define_SE3(
            lie_alg.exp_so3(np.array([1, 0, 0]) * (np.pi / 2)), [0, 0, 0.333 + 0.316 + 0.384])
        self.M[7] = lie_alg.define_SE3(
            np.dot(
                lie_alg.exp_so3(
                    np.array([1, 0, 0]) * np.pi), lie_alg.exp_so3(np.array([0, 0, 1]) * (np.pi / 4))), [0.088, 0, 0.333 + 0.316 + 0.384])

        # self.M_gripper = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0, 0]) * (np.pi)), [0.088, 0, 0.333 + 0.316 + 0.384 - 0.107 - 0.008])
        self.M_gripper = np.array(
             [[ 7.07106767e-01,  7.07106795e-01,  3.17476030e-19, 8.80000000e-02],
              [ 7.07106795e-01, -7.07106767e-01, -7.76501696e-19, -4.23913520e-20],
              [-3.24580175e-19,  7.73559062e-19, -1.00000000e+00, 9.26000007e-01],
              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
         
        self.gripper = Gripper(self.M_gripper, pcd=pcd, root=root) 

        if azure:
            # bracket
            bracket = o3d.io.read_triangle_mesh(os.path.join(root, f"robot_vis/mesh/bracket.ply"))
            bracket.vertices = o3d.utility.Vector3dVector(np.asarray(bracket.vertices) / 1000)
            bracket.compute_vertex_normals()
            bracket.paint_uniform_color(self.bracket_color)
            bracket.translate([- 0.063 / 2 , - 0.113 /2, - 0.052])

            # azure
            azure = o3d.io.read_triangle_mesh(os.path.join(root, f"robot_vis/mesh/Azure_Kinect.ply"))
            azure.vertices = o3d.utility.Vector3dVector(np.asarray(azure.vertices) / 1000)
            azure.compute_vertex_normals()
            azure.paint_uniform_color(self.azure_color)
            azure.transform(lie_alg.define_SE3(lie_alg.exp_so3(np.array([0, 0, 1]) * - np.pi / 2), [0.075 - 0.039 + 0.007, 0.103 / 2 , 0.03 - 0.01365 - 0.069]))

            # append to link
            self.link[8] = bracket
            self.link[9] = azure

            # azure
            self.M[8] = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0, 0]) * np.pi), [0.088, 0, 0.333 + 0.316 + 0.384 - 0.107])
            self.M[9] = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0, 0]) * np.pi), [0.088, 0, 0.333 + 0.316 + 0.384 - 0.107])

        self.link_SE3 = self.M.copy()

        if pcd:
            for link_num in range(n_link):
                if link_num <= 4:
                    self.link[link_num] = self.link[link_num].sample_points_uniformly(number_of_points=50)
                else:
                    self.link[link_num] = self.link[link_num].sample_points_uniformly(number_of_points=300)

            if self.mesh_bottle is not None:
                self.mesh_bottle = self.mesh_bottle.sample_points_uniformly(number_of_points=50)

        for link_num in range(n_link):
            self.link[link_num].transform(self.M[link_num])
        
        self.screw()
        self.set_joint_limit()
        self.q = np.zeros((7, ))

        if azure:
            self.mesh = [self.link[0], 
                        self.link[1], 
                        self.link[2], 
                        self.link[3], 
                        self.link[4], 
                        self.link[5], 
                        self.link[6], 
                        self.link[7], 
                        self.link[8], 
                        self.link[9]] + self.gripper.mesh
                    
        else:
            self.mesh = [self.link[0], 
                        self.link[1], 
                        self.link[2], 
                        self.link[3], 
                        self.link[4], 
                        self.link[5], 
                        self.link[6], 
                        self.link[7]] + self.gripper.mesh
            
        if add_bottle == True:
            self.mesh += [self.mesh_bottle]
        
        
    def screw(self):
        w_1 = np.array([0.0, 0.0, 1.0])
        w_2 = np.array([0.0, 1.0, 0.0])
        w_3 = np.array([0.0, 0.0, 1.0])
        w_4 = np.array([0.0, -1.0, 0.0])	
        w_5 = np.array([0.0, 0.0, 1.0])
        w_6 = np.array([0.0, -1.0, 0.0])
        w_7 = np.array([0.0, 0.0, -1.0])

        p_1 = np.array([0.0, 0.0, 0.333])
        p_2 = np.array([0.0, 0.0, 0.0])
        p_3 = np.array([0.0, 0.0, 0.316])
        p_4 = np.array([0.0825, 0.0, 0.0])
        p_5 = np.array([-0.0825, 0.0, 0.384])
        p_6 = np.array([0.0, 0.0, 0.0])
        p_7 = np.array([0.088, 0.0, 0.0])

        q_1 = p_1
        q_2 = p_1 + p_2
        q_3 = p_1 + p_2 + p_3
        q_4 = p_1 + p_2 + p_3 + p_4
        q_5 = p_1 + p_2 + p_3 + p_4 + p_5
        q_6 = p_1 + p_2 + p_3 + p_4 + p_5 + p_6
        q_7 = q_6 + p_7

        S = np.zeros((6,7))
        S[:,0] = np.concatenate((w_1, -np.cross(w_1, q_1)))
        S[:,1] = np.concatenate((w_2, -np.cross(w_2, q_2)))
        S[:,2] = np.concatenate((w_3, -np.cross(w_3, q_3)))
        S[:,3] = np.concatenate((w_4, -np.cross(w_4, q_4)))
        S[:,4] = np.concatenate((w_5, -np.cross(w_5, q_5)))
        S[:,5] = np.concatenate((w_6, -np.cross(w_6, q_6)))
        S[:,6] = np.concatenate((w_7, -np.cross(w_7, q_7)))

        self.screws = S


    def set_joint_limit(self, lb=None, ub=None):
        if not lb:
            lb = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        if not ub:
            ub = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        self.joint_limit = dict()
        self.joint_limit['lower_bound'] = lb
        self.joint_limit['upper_bound'] = ub
    
    def move_to_q(self, q_desired):
        for joint in range(len(q_desired)):
            T = forward_kinematics(self.screws[:, :joint + 1], q_desired[:joint + 1], self.M[joint + 1])
            self.link[joint + 1].transform(np.dot(T, lie_alg.inverse_SE3(self.link_SE3[joint + 1])))
        
            self.link_SE3[joint + 1] = T
        if self.azure:
            for camera_module in range(8, 10):
                T = forward_kinematics(self.screws, q_desired, self.M[camera_module])
                self.link[camera_module].transform(np.dot(T, lie_alg.inverse_SE3(self.link_SE3[camera_module])))
                self.link_SE3[camera_module] = T
            
        T = forward_kinematics(self.screws, q_desired, self.M_gripper)
        T_ee = forward_kinematics(self.screws, q_desired, self.franka_sub.initialEEFrame)
        # print(T)
        self.gripper.move_to_SE3(T)
        if self.mesh_bottle is not None:
            self.mesh_bottle.transform(
                
                    np.dot(traj_robotbase_EE_to_bottle_numpy_single(T_ee), lie_alg.inverse_SE3(self.bottle_SE3)))
            self.bottle_SE3 = traj_robotbase_EE_to_bottle_numpy_single(T_ee)

        self.q = q_desired

    def move_to_SE3(self, T_desired):
        q_init = np.mean((self.joint_limit['lower_bound'], self.joint_limit['upper_bound']), axis=0)
        q_desired = inverse_kinematics(self.screws, q_init, self.M[7], T_desired, self.joint_limit)
        self.move_to_q(q_desired)
        
    def set_finger_width(self, width):
        self.gripper.set_finger_width(width)

    def get_pcd_points(self, long=False):
        if not self.pcd:
            print('The mesh is not a pcd element')
            return np.array([])
        else:
            points = []
            for link_num in range(self.n_link):
                link = self.link[link_num]
                points.append(np.asarray(link.points))
            for i in range(3):
                link = self.gripper.mesh[i]
                points.append(np.asarray(link.points))
            if self.mesh_bottle is not None:
                points.append(np.asarray(self.mesh_bottle.points))
            if long:
                points = np.concatenate(points, axis=0)
                    
        return points

    def get_EE_T(self, q):
        _, T = Kinematics_numpy.forward_kinematics(q, self.franka_sub.S_screw, self.franka_sub.initialEEFrame)
        return T

class Gripper:
    def __init__(self, SE3, width=0, pcd=False, root=''):
            
        # initialize
        self.hand_SE3 = SE3
        self.gripper_width = width
        if width < 0:
            print("gripper width exceeds minimum width. gripper width is set to 0")
            self.gripper_width = 0
        if width > 0.08:
            print("gripper width exceeds maximum width. gripper width is set to 0.08")
            self.gripper_width = 0.08

        # 
        self.hand = o3d.io.read_triangle_mesh(os.path.join(root, "robot_vis/mesh/hand.ply"))
        self.hand.compute_vertex_normals()
        self.hand.paint_uniform_color([0.9, 0.9, 0.9])
        self.finger1 = o3d.io.read_triangle_mesh(os.path.join(root, "robot_vis/mesh/finger.ply"))
        self.finger1.compute_vertex_normals()
        self.finger1.paint_uniform_color([0.7, 0.7, 0.7])
        self.finger2 = o3d.io.read_triangle_mesh(os.path.join(root, "robot_vis/mesh/finger.ply"))
        self.finger2.compute_vertex_normals()
        self.finger2.paint_uniform_color([0.7, 0.7, 0.7])

        if pcd:
            self.hand = self.hand.sample_points_uniformly(number_of_points=200)
            self.finger1 = self.finger1.sample_points_uniformly(number_of_points=50)
            self.finger2 = self.finger2.sample_points_uniformly(number_of_points=50)
        self.finger1_M = lie_alg.define_SE3(np.identity(3), np.array([0, self.gripper_width/2, 0.1654/3]))
        self.finger2_M = lie_alg.define_SE3(lie_alg.exp_so3(np.asarray([0, 0, 1]) * np.pi), np.array([0, -self.gripper_width/2, 0.1654/3]))

        self.finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
        self.finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)
        
        self.hand.transform(self.hand_SE3)
        self.finger1.transform(self.finger1_SE3)
        self.finger2.transform(self.finger2_SE3)
        self.mesh = [self.hand, self.finger1, self.finger2]

    def move_to_SE3(self, desired_SE3):
        self.hand.transform(np.dot(desired_SE3, lie_alg.inverse_SE3(self.hand_SE3)))
        self.finger1.transform(np.dot(desired_SE3, lie_alg.inverse_SE3(self.hand_SE3)))
        self.finger2.transform(np.dot(desired_SE3, lie_alg.inverse_SE3(self.hand_SE3)))
        self.hand_SE3 = desired_SE3.copy()
        self.finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
        self.finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)
            
    def set_finger_width(self, width):
        self.gripper_width = width
        self.finger1_M = lie_alg.define_SE3(np.identity(3), np.array([0, self.gripper_width/2, 0.1654/3]))
        self.finger2_M = lie_alg.define_SE3(lie_alg.exp_so3(np.asarray([0, 0, 1]) * np.pi), np.array([0, -self.gripper_width/2, 0.1654/3]))

        desired_finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
        desired_finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)

        self.finger1.transform(np.dot(desired_finger1_SE3, lie_alg.inverse_SE3(self.finger1_SE3)))
        self.finger2.transform(np.dot(desired_finger2_SE3, lie_alg.inverse_SE3(self.finger2_SE3)))

        self.finger1_SE3 = desired_finger1_SE3.copy()
        self.finger2_SE3 = desired_finger2_SE3.copy()
            
    def add_to_vis(self, vis):
        vis.add_geometry(self.hand)
        vis.add_geometry(self.finger1)
        vis.add_geometry(self.finger2)
