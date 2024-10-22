import numpy as np
import os, sys, time

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from utils.lie import *
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

class Panda:
    def __init__(self, root='', T_base=np.eye(4), T_ee=np.eye(4), hand=True):
        
        self.n_dof = 7
        self.T_base = T_base
        self.hand = hand

        # screws A_i, i-th screw described in i-th frame
        self.A = np.array([ [0, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0], 
                            [1, 1, 1, 1, 1, 1, 1], 
                            [0, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0.0]]) 
        
        # M : M_i == link frames M_{i,i-1}
        self.M = np.zeros((4, 4, self.n_dof))
        self.M[:, :, 0] = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 1, 0.333], 
                                    [0, 0, 0, 1]])
        
        self.M[:, :, 1] = np.array([[1, 0, 0, 0], 
                                    [0, 0, 1, 0], 
                                    [0,-1, 0, 0], 
                                    [0, 0, 0, 1.0]])
        
        self.M[:, :, 2] = np.array([[1, 0, 0, 0], 
                                    [0, 0, -1, -0.316], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 0, 1]])
        
        self.M[:, :, 3] = np.array([[1, 0, 0, 0.0825], 
                                    [0, 0,-1, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 0, 1]])
        
        self.M[:, :, 4] = np.array([[1, 0, 0, -0.0825], 
                                    [0, 0, 1, 0.384], 
                                    [0,-1, 0, 0], 
                                    [0, 0, 0, 1]])
            
        self.M[:, :, 5] = np.array([[1, 0, 0, 0], 
                                    [0, 0,-1, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 0, 1.0]])
        
        self.M[:, :, 6] = np.array([[1, 0, 0, 0.088], 
                                    [0, 0,-1, 0], 
                                    [0, 1, 0, 0], 
                                    [0, 0, 0, 1]])
        
        for i in range(self.n_dof):
            self.M[:, :, i] = inv_SE3(self.M[:, :, i])
   
        self.M_ee = T_ee
        
        self.M_sb = np.eye(4)
        for i in range(self.n_dof):
            self.M_sb = self.M_sb @ inv_SE3(self.M[:, :, i])
        self.M_sb = self.M_sb @ self.M_ee
        
        # Spatial Screw A_s and Body Screw A_b
        self.A_s = np.zeros(self.A.shape)
        self.A_b = np.zeros(self.A.shape)
        M_si = np.eye(4)
        for i in range(self.n_dof):
            M_si = M_si @ inv_SE3(self.M[:, :, i])
            self.A_s[:, i] = Adjoint_SE3(M_si) @ self.A[:, i]
            self.A_b[:, i] = Adjoint_SE3(inv_SE3(self.M_sb)) @ self.A_s[:, i]
        
        self.A_bj = np.zeros((self.n_dof, 6, self.n_dof))
        for j in range(self.n_dof):
            M_sj = np.eye(4)
            for i in range(j+1):
                M_sj = M_sj @ inv_SE3(self.M[:, :, i])
            for i in range(self.n_dof):
                self.A_bj[j, :, i] = Adjoint_SE3(inv_SE3(M_sj)) @ self.A_s[:, i]
            
        self.q_min = np.array([[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]]).T
        self.q_max = np.array([[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.752, 2.8973]]).T
        self.qdot_min = np.array([[-2.1750, -2.1750, -2.1750, -2.1750, -2.6100, -2.6100, -2.6100]]).T
        self.qdot_max = np.array([[2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]]).T
        
        self.tau_max = np.array([[87, 87, 87, 87, 12, 12, 12]]).T
        self.tau_min = self.tau_max * -1
        self.dtau_max = np.array([[1000, 1000, 1000, 1000, 1000, 1000, 1000]]).T
        
        self.Vdot0 = np.array([0, 0, 0, 0, 0, 9.8])
        
        self.points = []
        for i in range(self.n_dof+1):
            self.points.append(np.load(os.path.join(root, f'assets/panda/points/link{i}.npy')))
        
        if hand:
            self.points.append(np.load(os.path.join(root, f'assets/panda/points/hand.npy')))
        
        # Current states
        self.q = None
        self.qdot = None
        self.T_sb = None
        self.T_links = None
        self.Jb = None
        self.Js = None
        self.Ja = None
        self.Jb_link = None
        self.dTdq = None
        
        self.setRobotState(np.zeros(self.n_dof))
        
    def setRobotState(self, jointPos, jointVel=None):
        if jointVel is None:
            jointVel = np.zeros(self.n_dof)
        self.q = np.asarray(jointPos)
        self.qdot = np.asarray(jointVel)
        T_sb, T_links = self.solveForwardKinematics(jointPos, return_T_link=True)
        self.T_sb = T_sb
        self.T_links = T_links
        self.Jb = self.getBodyJacobian(jointPos)
        # self.Js = self.getSpatialJacobian(jointPos)
        self.Ja = self.getAnalyticJacobian(jointPos, J_b=self.Jb, T_sb=self.T_sb)
        self.Jb_link = self.getLinkBodyJacobian(jointPos)
        self.dTdq = self.getdTdq(jointPos, Jb_js=self.Jb_link, T_links=self.T_links)

    def solveForwardKinematics(self, jointPos, jointVel=None, jointAcc=None, return_T_link=False):
        assert len(jointPos) == self.n_dof
        
        if jointVel is None:
            jointVel = np.zeros(self.n_dof)
        if jointAcc is None:
            jointAcc = np.zeros(self.n_dof)
            
        M_ = np.zeros((self.n_dof, 4, 4))
        for i in range(self.n_dof):
            M_[i, :, :] = inv_SE3(self.M[:, :, i])
        
        M_sb = np.eye(4)
        T_sj = np.zeros((self.n_dof, 4, 4))
        for i in range(self.n_dof):
            M_sb = M_sb @ M_[i, :, :] @ exp_se3(self.A[:, i]*jointPos[i])
            T_sj[i] = self.T_base @ M_sb
            
        M_sb = M_sb @ self.M_ee
        T_sb = self.T_base @ M_sb
        if return_T_link:
            return T_sb, T_sj
        else:
            return T_sb
    
    def solveInverseKinematics(
        self, 
        T_desired, 
        init_jointPos, eps=1e-5, step_size=1, 
        max_iter=100, jointLowerLimits=None, 
        jointUpperLimits=None, desired_q=None):
        assert len(init_jointPos) == self.n_dof

        jointPos = init_jointPos
        T = self.solveForwardKinematics(jointPos)
        V = invskew(log_SE3(inv_SE3(T) @ T_desired))
        error = np.linalg.norm(V)
        converge_flag = False
        for _ in range(max_iter):
            T = self.solveForwardKinematics(jointPos)
            V = invskew(log_SE3(inv_SE3(T) @ T_desired))
            error = np.linalg.norm(V)
            if error < eps:
                converge_flag = True
                break
            else:
                J = self.getBodyJacobian(jointPos)
                J_pinv = np.linalg.pinv(J)
                qdot = J_pinv @ V
                jointPos = jointPos + qdot * step_size
                if desired_q is not None:
                    vec = (np.eye(7) - J_pinv@J)@(desired_q - jointPos)
                    jointPos = jointPos + vec * step_size
                
            if jointLowerLimits is not None:
                violation_index = jointPos < jointLowerLimits + 0.01
                jointPos[violation_index] = jointLowerLimits[violation_index] + 0.01
            if jointUpperLimits is not None:
                violation_index = jointPos > jointUpperLimits - 0.01
                jointPos[violation_index] = jointUpperLimits[violation_index] - 0.01
            
                
        if converge_flag:
            return converge_flag, jointPos
        else:
            return converge_flag, init_jointPos
        
    def solvePointInverseKinematics(self, p_desired, init_jointPos, eps=1e-5, step_size=1, max_iter=100):
        assert len(init_jointPos) == self.n_dof
        
        p_desired = np.asarray(p_desired)
        
        jointPos = init_jointPos
        T = self.solveForwardKinematics(jointPos)
        V = p_desired - T[:3, 3]
        error = np.linalg.norm(V)
        converge_flag = False
        for _ in range(max_iter):
            T = self.solveForwardKinematics(jointPos)
            V = V = p_desired - T[:3, 3]
            error = np.linalg.norm(V)
            if error < eps:
                converge_flag = True
                break
            else:
                Ja = self.getAnalyticJacobian(jointPos)
                qdot = np.linalg.pinv(Ja[3:]) @ V
                jointPos = jointPos + qdot * step_size
        
        if converge_flag:
            return jointPos
        else:
            return init_jointPos
    
    def getBodyJacobian(self, jointPos):
        assert len(jointPos) == self.n_dof
        T = np.eye(4)
        J_b = np.zeros((6, self.n_dof))
        for i in range(self.n_dof-1, -1, -1):
            J_b[:, i] = Adjoint_SE3(T) @ self.A_b[:, i]
            T = T @ exp_se3(-self.A_b[:, i]*jointPos[i])
        return J_b
    
    def getLinkBodyJacobian(self, jointPos):
        assert len(jointPos) == self.n_dof
        J_b = np.zeros((self.n_dof, 6, self.n_dof))
        for j in range(self.n_dof):
            T = np.eye(4)
            for i in range(j, -1, -1):
                J_b[j, :, i] = Adjoint_SE3(T) @ self.A_bj[j, :, i]
                T = T @ exp_se3(-self.A_bj[j, :, i]*jointPos[i])
        return J_b
    
    def getdTdq(self, jointPos, Jb_js=None, T_links=None):
        # output : (link idx, 4, 4, joint idx) == (DOF, 4, 4, DOF)
        assert len(jointPos) == self.n_dof
        
        if Jb_js is None:
            Jb_js = self.getLinkBodyJacobian(jointPos)
        
        if T_links is None:
            _, T_links = self.solveForwardKinematics(jointPos, return_T_link=True)
        
        bracket_Jb_js = np.zeros((7, 4, 4, 7))
        bracket_Jb_js[:, 0, 1, :] = -Jb_js[:, 2, :]
        bracket_Jb_js[:, 0, 2, :] = Jb_js[:, 1, :]
        bracket_Jb_js[:, 0, 3, :] = Jb_js[:, 3, :]
        bracket_Jb_js[:, 1, 0, :] = Jb_js[:, 2, :]
        bracket_Jb_js[:, 1, 2, :] = -Jb_js[:, 0, :]
        bracket_Jb_js[:, 1, 3, :] = Jb_js[:, 4, :]
        bracket_Jb_js[:, 2, 0, :] = -Jb_js[:, 1, :]
        bracket_Jb_js[:, 2, 1, :] = Jb_js[:, 0, :]
        bracket_Jb_js[:, 2, 3, :] = Jb_js[:, 5, :]
        
        dTdq = np.einsum('bij, bjkl -> bikl', T_links, bracket_Jb_js)
            
        return dTdq
        
    def getSpatialJacobian(self, jointPos):
        assert len(jointPos) == self.n_dof

        T = np.eye(4)
        J_s = np.zeros((6, self.n_dof))
        for i in range(self.n_dof):
            J_s[:, i] = Adjoint_SE3(T) @ self.A_s[:, i]
            T = T @ exp_se3((self.A_s[:, i]*jointPos[i]))
        return J_s
    
    def getAnalyticJacobian(self, jointPos, J_b=None, T_sb=None):
        assert len(jointPos) == self.n_dof
        
        if J_b is None:
            J_b = self.getBodyJacobian(jointPos)
        if T_sb is None:
            T_sb = self.solveForwardKinematics(jointPos)
        
        bigR_sb = np.eye(6)
        bigR_sb[:3, :3] = T_sb[:3, :3]
        bigR_sb[3:, 3:] = T_sb[:3, :3]
        J_a = bigR_sb @ J_b
        return J_a
    
    def getPointCloud(self, jointPos):
        T_sb, T_links = self.solveForwardKinematics(jointPos, return_T_link=True)
        
        total_points = np.empty((0, 3))
        
        link0_points = self.T_base[:3, :3] @ self.points[0].T + self.T_base[:3, 3].reshape(3, 1).repeat(len(self.points[0]), 1)
        total_points = np.concatenate((total_points, link0_points.T), axis=0)
        
        for i in range(1, self.n_dof+1):
            linki_points = T_links[i-1, :3, :3] @ self.points[i].T + T_links[i-1, :3, 3].reshape(3, 1).repeat(len(self.points[i]), 1)
            total_points = np.concatenate((total_points, linki_points.T), axis=0)
        
        if self.hand:
            R_hand = Rot.from_euler('z', -0.785398163397)
            p_hand = np.array([0, 0, 0.107])
            T_hand = np.eye(4)
            T_hand[:3, :3] = R_hand.as_matrix()
            T_hand[:3, 3] = p_hand
            T_hand = T_links[-1] @ T_hand
            
            hand_points = T_hand[:3, :3] @ self.points[-1].T + T_hand[:3, 3].reshape(3, 1).repeat(len(self.points[-1]), 1)
            total_points = np.concatenate((total_points, hand_points.T), axis=0)
            
        return total_points
    
if __name__ == '__main__':
    robot = Panda()