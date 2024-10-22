from re import T
import numpy as np
import torch.nn as nn
import torch
from utils import LieGroup_torch as lie

class BaseGroup():
    def __init__(self) -> None:
        pass

    def project_group(self, tau):
        # return: g
        raise NotImplementedError

    def _check_valid(self, tol=1e-4):
        for i in range(50):
            g = self._random_g()
            g_inv = self.get_inv(g)
            tau = self._random_task()
            traj = self._random_traj()
            gtau, gtraj = self.action_traj(g, tau, traj)

            ghat_1 = self.project_group(tau)
            ghat_2 = self.project_group(gtau)

            tau0_1, traj0_1 = self.action_traj(ghat_1, tau, traj)
            tau0_2, traj0_2 = self.action_traj(ghat_2, gtau, gtraj)
            tau0_error = torch.norm(tau0_1 - tau0_2)
            traj0_error = torch.norm(traj0_1 - traj0_2, dim=1).mean()
            g_inv_error = torch.norm(tau - self.action_task(g_inv, self.action_task(g, tau)))
            if tau0_error <= tol and traj0_error <= tol and g_inv_error <= tol:
                continue
            else:
                print(f'iteration = {i}')
                print('tau0_delta:')
                print(tau0_error)
                print('traj0_delta:')
                print(traj0_error)
                print('g_inv_error')
                print(g_inv_error)
                return False
        return True

    def _random_g(self, batchsize=10):
        raise NotImplementedError
    
    def _random_task(self, batchsize=10):
        raise NotImplementedError

    def _random_traj(self, batchsize=10):
        raise NotImplementedError

    def get_inv(self, g):
        raise NotImplementedError

    def action_task(self, g, tau):
        raise NotImplementedError

    def action_traj(self, g, tau, traj):
        raise NotImplementedError
    
    def action_manner(self, g, tau, z):
        return self.action_task(g, tau), z
    
    def _random_data_aug(self, tau, traj=None, n_aug=1):
        n_data = len(tau)
        tau_aug_list = []
        traj_aug_list = []
        if n_aug > 0:
            for i in range(n_aug):
                g = self._random_g(batch_size=n_data).to(tau)
                if traj is not None:
                    gtau, gtraj = self.action_traj(g, tau, traj)
                    traj_aug_list.append(gtraj)
                else:
                    gtau = self.action_task(g, tau)
                tau_aug_list.append(gtau)
            tau_aug = torch.cat(tau_aug_list, dim=0)
            if traj is not None:
                traj_aug = torch.cat(traj_aug_list, dim=0)
        else:
            tau_aug = tau
            traj_aug = traj
        if traj is not None:
            return tau_aug, traj_aug
        else:
            return tau_aug
    
class PouringGroup(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_cup, y_cup, x_bottle, y_bottle, water_height, \theta_i) (p_i, theta oo)
        # g = ([x, y], theta1(rotation around the cup),
        #       theta2(rotation around the z-axis of the end-effector))
        # traj: SE(3)^N_t
        self.w_dim_squeezed = 2
        self.w_dim_local = 6
        self.w_dim_extended = 7
        self.bottle_height_bottom = 0.2
    
    def _random_g(self, batch_size=10):
        random_x = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_theta1_cup = torch.rand(batch_size, 1) * 2 * np.pi
        random_theta2_bottle = torch.rand(batch_size, 1) * 2 * np.pi
        random_g = torch.cat([random_x, random_y, random_theta1_cup, random_theta2_bottle], dim=1)
        # print(f'random_g = {random_g.dtype}')
        return random_g
    
    def _random_task(self, batch_size=10):
        random_x_cup = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y_cup = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_x_i = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y_i = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_h = torch.rand(batch_size, 1) * 0.21 + 0.2
        random_theta_i= torch.rand(batch_size, 1) * 2 * torch.pi
        random_tau = torch.cat([
            random_x_cup, random_y_cup, random_x_i, random_y_i, random_h, random_theta_i
        ], dim=1)
        return random_tau

    def _random_traj(self, batch_size = 10):
        S = (torch.rand(batch_size, 100, 6)*10) - 5
        Slong = S.reshape(-1, 6)
        Tlong = lie.exp_se3(Slong)
        T = Tlong.reshape(batch_size, 100, 4, 4)
        return T
    
    # squeeze_hat_w and unsqueeze_hat_w : for reduced task parameter learning of equivariant_TCVAE
    def squeeze_hat_w(self, hat_w):
        return hat_w[:, [2, 4]]
    
    def unsqueeze_hat_w(self, squeezed_hat_w):
        batch_size = len(squeezed_hat_w)
        temp = torch.zeros((batch_size, 1))
        return torch.cat([temp, temp, squeezed_hat_w[:, 0], temp, squeezed_hat_w[:, 1], temp], dim=1)
    
    # extended_w and unextended_w : for extended task parameter learning of TCVAE
    def extend_w(self, w):
        theta = w[:, 5]
        cos_theta = torch.cos(theta).unsqueeze(1); sin_theta = torch.sin(theta).unsqueeze(1)

        return torch.cat((w[:, :5], cos_theta, sin_theta), dim=1)
    
    def unextend_w(self, extended_w):
        theta = torch.atan2(extended_w[:, 6], extended_w[:, 5]).unsqueeze(1)
        
        return torch.cat((extended_w[:, :5], theta), dim=1)
    
    def project_group(self, tau):
        batch_size = len(tau)
        g = torch.zeros(batch_size, 4).to(tau)
        g[:, :2] = -tau[:, :2]
        d_xy = tau[:, 2:4] - tau[:, :2]
        theta_1 = torch.atan2(d_xy[:, 1], d_xy[:, 0])
        theta_2 = tau[:, -1]
        g[:, 2] = -theta_1
        g[:, 3] = -theta_2 + theta_1
        return g
        
    def get_inv(self, g):
        return -g
    
    def rot_z(self, theta, SE3=True):
        batch_size = len(theta)
        w = torch.zeros(batch_size, 3).to(theta)
        w[:, 2] = theta
        R = lie.exp_so3(w)
        if SE3 == False:
            return R
        else:
            T = torch.cat(
                [torch.cat([R, torch.zeros(batch_size, 3, 1).to(theta)], dim=2), 
                torch.zeros(batch_size, 1, 4).to(theta)
                ], dim=1)
            T[:, -1, -1] = 1
            return T

    def action_task(self, g, tau):
        batch_size = len(g)
        gtau = torch.zeros_like(tau)
        gtau[:, :2] = tau[:, :2] + g[:, :2]
        theta_1 = g[:, 2]
        theta_2 = g[:, 3]
        d_x = tau[:, 2] - tau[:, 0]
        d_y = tau[:, 3] - tau[:, 1]
        c1 = torch.cos(theta_1)
        s1 = torch.sin(theta_1)
        gtau[:, 2] = tau[:, 0] + g[:, 0] + (c1 * d_x - s1 * d_y)
        gtau[:, 3] = tau[:, 1] + g[:, 1] + (s1 * d_x + c1 * d_y)
        gtau[:, 4] = tau[:, 4]
        gtau[:, 5] = tau[:, 5] + theta_1 + theta_2
        return gtau

    def action_traj(self, g, tau, traj):
        original_shape = traj.shape
        dim12 = False
        if len(original_shape) == 2:
            dim12 = True
            time_step = int(traj.shape[-1] / 12)
        else:
            time_step = traj.shape[1]
        batch_size = len(g)
        traj = traj.reshape(batch_size, time_step, -1, 4)
        if traj.size()[2] == 3:
            aug = torch.zeros((batch_size, time_step, 1, 4)).to(traj)
            aug[:, :, :, 3] = 1
            traj = torch.cat((traj, aug), dim=2)
            dim12 = True
        gtau = self.action_task(g, tau)
        theta_1 = g[:, 2]
        theta_2 = g[:, 3]
        S = torch.zeros(batch_size, 6).to(traj)
        S[:, 2] = 1
        S[:, 3] = tau[:, 1]
        S[:, 4] = -tau[:, 0]
        T_screw = lie.exp_se3(S * theta_1.unsqueeze(1))
        traj_after_screw_motion = torch.einsum('ijk, ihkl -> ihjl', T_screw, traj)
        gtraj = torch.einsum('ijkl, ilh -> ijkh', traj_after_screw_motion, self.rot_z(theta_2))
        # gtraj = T_screw.unsqueeze(1) @ traj @ self.rot_z(theta_2)
        gtraj[:, :, 0, -1] += g[:, 0].unsqueeze(1)
        gtraj[:, :, 1, -1] += g[:, 1].unsqueeze(1)
        if dim12:
            gtraj = gtraj[:, :, :3, :].reshape(*original_shape)
        return gtau, gtraj

class PouringGroup_old(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_cup, y_cup, x_i, y_i, z_i, \theta_i) (p_i, theta oo)
        # g = ([x, y], theta1(rotation around the cup),
        #       theta2(rotation around the z-axis of the end-effector))
        # traj: SE(3)^N_t
        self.w_dim_squeezed = 2
        self.w_dim_local = 6
        self.w_dim_extended = 7
    
    def _random_g(self, batch_size=10):
        random_x = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_theta1_cup = torch.rand(batch_size, 1) * 2 * np.pi
        random_theta2_bottle = torch.rand(batch_size, 1) * 2 * np.pi
        random_g = torch.cat([random_x, random_y, random_theta1_cup, random_theta2_bottle], dim=1)
        return random_g
    
    def _random_task(self, batch_size=10):
        random_x_cup = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y_cup = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_x_i = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_y_i = torch.rand(batch_size, 1) * 0.5 - 0.25
        random_z_i = torch.rand(batch_size, 1) * 0.2 + 0.4
        random_theta_i= torch.rand(batch_size, 1) * 2 * torch.pi
        random_tau = torch.cat([
            random_x_cup, random_y_cup, random_x_i, random_y_i, random_z_i, random_theta_i
        ], dim=1)
        return random_tau

    def _random_traj(self, batch_size = 10):
        S = (torch.rand(batch_size, 100, 6)*10) - 5
        Slong = S.reshape(-1, 6)
        Tlong = lie.exp_se3(Slong)
        T = Tlong.reshape(batch_size, 100, 4, 4)
        return T
    
    
    # squeeze_hat_w and unsqueeze_hat_w : for reduced task parameter learning of equivariant_TCVAE
    def squeeze_hat_w(self, hat_w):
        return hat_w[:, [2, 4]]
    
    def unsqueeze_hat_w(self, squeezed_hat_w):
        batch_size = len(squeezed_hat_w)
        temp = torch.zeros((batch_size, 1))
        return torch.cat([temp, temp, squeezed_hat_w[:, 0], temp, squeezed_hat_w[:, 1], temp], dim=1)
    
    # extended_w and unextended_w : for extended task parameter learning of TCVAE
    def extend_w(self, w):
        theta = w[:, 5]
        cos_theta = torch.cos(theta).unsqueeze(1); sin_theta = torch.sin(theta).unsqueeze(1)

        return torch.cat((w[:, :5], cos_theta, sin_theta), dim=1)
    
    def unextend_w(self, extended_w):
        theta = torch.atan2(extended_w[:, 6], extended_w[:, 5]).unsqueeze(1)
        
        return torch.cat((extended_w[:, :5], theta), dim=1)
    
    def project_group(self, tau):
        batch_size = len(tau)
        g = torch.zeros(batch_size, 4).to(tau)
        g[:, :2] = -tau[:, :2]
        d_xy = tau[:, 2:4] - tau[:, :2]
        theta_1 = torch.atan2(d_xy[:, 1], d_xy[:, 0])
        theta_2 = tau[:, -1]
        g[:, 2] = -theta_1
        g[:, 3] = -theta_2 + theta_1
        return g
        
    def get_inv(self, g):
        return -g
    
    def rot_z(self, theta, SE3=True):
        batch_size = len(theta)
        w = torch.zeros(batch_size, 3).to(theta)
        w[:, 2] = theta
        R = lie.exp_so3(w)
        if SE3 == False:
            return R
        else:
            T = torch.cat(
                [torch.cat([R, torch.zeros(batch_size, 3, 1).to(theta)], dim=2), 
                torch.zeros(batch_size, 1, 4).to(theta)
                ], dim=1)
            T[:, -1, -1] = 1
            return T

    def action_task(self, g, tau):
        batch_size = len(g)
        gtau = torch.zeros_like(tau)
        gtau[:, :2] = tau[:, :2] + g[:, :2]
        theta_1 = g[:, 2]
        theta_2 = g[:, 3]
        d_x = tau[:, 2] - tau[:, 0]
        d_y = tau[:, 3] - tau[:, 1]
        c1 = torch.cos(theta_1)
        s1 = torch.sin(theta_1)
        gtau[:, 2] = tau[:, 0] + g[:, 0] + (c1 * d_x - s1 * d_y)
        gtau[:, 3] = tau[:, 1] + g[:, 1] + (s1 * d_x + c1 * d_y)
        gtau[:, 4] = tau[:, 4]
        gtau[:, 5] = tau[:, 5] + theta_1 + theta_2
        return gtau

    def action_traj(self, g, tau, traj):
        batch_size = len(g)
        time_step = traj.size()[1]
        traj = traj.reshape(batch_size, time_step, 3, 4)
        if traj.size()[2] == 3:
            aug = torch.zeros((batch_size, time_step, 1, 4)).to(traj)
            aug[:, :, :, 3] = 1
            traj = torch.cat((traj, aug), dim=2)
        
        gtau = self.action_task(g, tau)
        theta_1 = g[:, 2]
        theta_2 = g[:, 3]
        S = torch.zeros(batch_size, 6).to(traj)
        S[:, 2] = 1
        S[:, 3] = tau[:, 1]
        S[:, 4] = -tau[:, 0]
        T_screw = lie.exp_se3(S * theta_1.unsqueeze(1))
        traj_after_screw_motion = torch.einsum('ijk, ihkl -> ihjl', T_screw, traj)
        gtraj = torch.einsum('ijkl, ilh -> ijkh', traj_after_screw_motion, self.rot_z(theta_2))
        # gtraj = T_screw.unsqueeze(1) @ traj @ self.rot_z(theta_2)
        gtraj[:, :, 0, -1] += g[:, 0].unsqueeze(1)
        gtraj[:, :, 1, -1] += g[:, 1].unsqueeze(1)
        return gtau, gtraj[:, :, :3, :].reshape(batch_size, time_step, -1)

class PouringGroup_single(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_cup, y_cup, x_i, y_i, z_i, \theta_i) (p_i, theta oo)
        # g = ([x, y], theta1(rotation around the cup),
        #       theta2(rotation around the z-axis of the end-effector))
        # traj: SE(3)^N_t
    
    def _random_g(self, batch_size=10):
        random_rot = torch.randint(0, 4, (batch_size, 1))
        random_mirror = torch.randint(0, 2, (batch_size, 1))
        random_g = torch.cat([random_rot, random_mirror], dim=1)
        return random_g
    
    def _random_task(self, batch_size=10):
        random_theta = np.random.random() * 2 * np.pi
        random_r = np.sqrt(np.random.random()) * 5 + 5
        random_x = np.cos(random_theta) * random_r
        random_y = np.sin(random_theta) * random_r
        random_tau = torch.cat([
            random_x, random_y,
        ], dim=1)
        return random_tau

    def _random_traj(self, batch_size=10):
        traj = (torch.rand(batch_size, 100, 2)*10)
        return traj
    # squeeze_hat_w and unsqueeze_hat_w : for reduced task parameter learning of equivariant_TCVAE
    def squeeze_hat_w(self, hat_w):
        return hat_w[:, [2, 4]]
    
    def unsqueeze_hat_w(self, squeezed_hat_w):
        batch_size = len(squeezed_hat_w)
        temp = torch.zeros((batch_size, 1))
        return torch.cat((temp, temp, squeezed_hat_w[:, 0], temp, squeezed_hat_w[:, 1], temp), dim=1)
    
    # extended_w and unextended_w : for extended task parameter learning of TCVAE
    def extend_w(self, w):
        theta = w[:, 5]
        cos_theta = torch.cos(theta).unsqueeze(1); sin_theta = torch.sin(theta).unsqueeze(1)

        return torch.cat((w[:, :5], cos_theta, sin_theta), dim=1)
    
    def unextend_w(self, extended_w):
        theta = torch.atan2(extended_w[:, 5], extended_w[:, 6]).unsqueeze(1)
        
        return torch.cat((extended_w[:, :5], theta), dim=1)
    
    def project_group(self, tau):
        batch_size = len(tau)
        g_rot = torch.zeros(batch_size).to(tau)
        g_mirror = torch.zeros(batch_size).to(tau)
        theta = torch.atan2(tau[:, 1], tau[:, 0])
        r = torch.norm(tau, dim=1)
        pi4 = torch.pi/4
        g_rot[(theta >= -pi4) * (theta < pi4)] = 0
        g_rot[(theta >= pi4) * (theta < 2 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 1
        g_rot[(theta >= 3* pi4) * (theta <= 4 * pi4) + (theta >= -4 * pi4) * (theta < -3 * pi4)] = 2
        g_rot[(theta >= 2* pi4) * (theta < 3 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 3
        
        g_mirror[(theta >= 0) * (theta < pi4) + (theta >= 2 * pi4) * (theta < 3 * pi4) + 
           (theta >= -4 * pi4) * (theta < -3 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 0
        g_mirror[(theta >= -pi4) * (theta < 0) + (theta >= pi4) * (theta < 2 * pi4) + 
           (theta >= 3 * pi4) * (theta <= 4 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 1
        g = torch.cat([g_rot.unsqueeze(1), g_mirror.unsqueeze(1)], dim=1)
        return g
        
    def get_inv(self, g):
        g[:, 0] = -g[:, 0] % 4
        return g

    def rot_matrix(self, theta):
        theta = torch.tensor(theta)
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.tensor([[c, -s], [s, c]])
        return
    
    def action_task(self, g, tau):
        # mirroring first, rotation later
        batch_size = len(g)
        g_rot = g[:, 0]
        g_mirror = g[:, 1]
        pi2 = torch.pi / 2
        gtau = tau.clone()
        gtau[g_mirror == 1][:, 1] = -gtau[g_mirror == 1][:, 1]
        #################################################
        
        R_mul = torch.zeros(batch_size, 2, 2)
        R_mul[g_rot == 0] = self.R[0]
        R_mul[g_rot == 1] = self.R[1]
        R_mul[g_rot == 2] = self.R[2]
        R_mul[g_rot == 3] = self.R[3]
        gtau = R_mul@gtau
        
        # gtau = torch.zeros_like(tau)
        # gtau[:, :2] = tau[:, :2] + g[:, :2]
        # theta_1 = g[:, 2]
        # theta_2 = g[:, 3]
        # d_x = tau[:, 2] - tau[:, 0]
        # d_y = tau[:, 3] - tau[:, 1]
        # c1 = torch.cos(theta_1)
        # s1 = torch.sin(theta_1)
        # gtau[:, 2] = tau[:, 0] + g[:, 0] + (c1 * d_x - s1 * d_y)
        # gtau[:, 3] = tau[:, 1] + g[:, 1] + (s1 * d_x + c1 * d_y)
        # gtau[:, 4] = tau[:, 4]
        # gtau[:, 5] = tau[:, 5] + theta_1 + theta_2
        return gtau

    def action_traj(self, g, tau, traj):
        batch_size = len(g)
        gtau = self.action_task(g, tau)
        theta_1 = g[:, 2]
        theta_2 = g[:, 3]
        S = torch.zeros(batch_size, 6).to(traj)
        S[:, 2] = 1
        S[:, 3] = tau[:, 1]
        S[:, 4] = -tau[:, 0]
        T_screw = lie.exp_se3(S * theta_1.unsqueeze(1))
        traj_after_screw_motion = torch.einsum('ijk, ihkl -> ihjl', T_screw, traj)
        gtraj = torch.einsum('ijkl, ilh -> ijkh', traj_after_screw_motion, self.rot_z(theta_2))
        # gtraj = T_screw.unsqueeze(1) @ traj @ self.rot_z(theta_2)
        gtraj[:, :, 0, -1] += g[:, 0].unsqueeze(1)
        gtraj[:, :, 1, -1] += g[:, 1].unsqueeze(1)
        return gtau, gtraj

class PouringGroupSingleTraj(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_cup, y_cup, x_i, y_i, z_i, \theta_i) (p_i, theta oo)
        # g = ([x, y], theta1(rotation around the cup),
        #       theta2(rotation around the z-axis of the end-effector))
        # traj: SE(3)^N_t
    
    def _random_g(self):
        return (torch.rand(4)*10) - 5
    
    def _random_task(self):
        return (torch.rand(6)*10) - 5

    def _random_traj(self):
        S = (torch.rand(100, 6)*10) - 5
        return lie.exp_se3(S)
    
    def project_group(self, tau):
        g = torch.zeros(4).to(tau)
        g[:2] = -tau[:2]
        d_xy = tau[2:4] - tau[:2]
        theta_1 = torch.atan2(d_xy[1], d_xy[0])
        theta_2 = tau[-1]
        g[2] = -theta_1
        g[3] = -theta_2 + theta_1
        return g
        
    def get_inv(self, g):
        return -g
    
    # def get_Ti_from_tau(self, tau, relative_to_cup=False):
    #     p = tau[2:5]
    #     if relative_to_cup == True:
    #         p[2:4] -= tau[:2]
    #     theta = tau[-1]
    #     w = torch.zeros(3).to(tau)
    #     w[-1] = theta
    #     wp = torch.cat([w, p]).unsqueeze(0)
    #     T = lie.exp_so3_T(wp).reshape(4,4)
    #     return T
    

    def rot_z(self, theta, SE3=True):
        w = torch.zeros(1, 3).to(theta)
        w[0, 2] = theta
        R = lie.exp_so3(w).squeeze(0)
        if SE3 == False:
            return R
        else:
            T = torch.cat(
                [torch.cat([R, torch.zeros(3, 1).to(theta)], dim=1), 
                torch.eye(4).to(theta)[-1:]
                ], dim=0)
            return T

    def action_task(self, g, tau):
        gtau = torch.zeros_like(tau)
        gtau[:2] = tau[:2] + g[:2]
        theta_1 = g[2]
        theta_2 = g[3]
        d_x = tau[2] - tau[0]
        d_y = tau[3] - tau[1]
        c1 = torch.cos(theta_1)
        s1 = torch.sin(theta_1)
        gtau[2] = tau[0] + g[0] + (c1 * d_x - s1 * d_y)
        gtau[3] = tau[1] + g[1] + (s1 * d_x + c1 * d_y)
        gtau[4] = tau[4]
        gtau[5] = tau[5] + theta_1 + theta_2
        return gtau

    def action_traj(self, g, tau, traj):
        gtau = self.action_task(g, tau)
        theta_1 = g[2]
        theta_2 = g[3]
        S = torch.zeros(1, 6).to(traj)
        S[0, 2] = 1
        S[0, 3] = tau[1]
        S[0, 4] = -tau[0]
        T_screw = lie.exp_se3(S * theta_1)
        gtraj = T_screw @ traj @ self.rot_z(theta_2).unsqueeze(0)
        gtraj[:, 0, -1] += g[0]
        gtraj[:, 1, -1] += g[1]
        return gtau, gtraj
################################################################################################
######################################### Toy 2D ###############################################
################################################################################################
class PlanarMobileRobot(BaseGroup):
    def __init__(self) -> None:
        super().__init__()
        # tau: a torch tensor (x_robot, y_robot, wall_rotation_angle)
        # G = {0, 1, 2 ,3} x {0, 1} x SO(2),
        # g = (rot, mirror, \theta_wall_robot)
        # traj: (R^2)^n
        self.R = torch.zeros(4, 2, 2)
        self.R[0, 0, 0] = 1
        self.R[0, 1, 1] = 1
        self.R[1, 0, 1] = -1
        self.R[1, 1, 0] = 1
        self.R[2, 0, 0] = -1
        self.R[2, 1, 1] = -1
        self.R[3, 0, 1] = 1
        self.R[3, 1, 0] = -1
        self.w_dim_squeezed = 2
        self.w_dim_local = 3
        self.w_dim_extended = 4
    
    def get_wall(self, tau):
        low = 1.2
        high = 4   
        # tau = (n, 3)
        wall1 = torch.tensor([[low, low, high], [high, low, low]])
        wall2 = torch.tensor([[low, low, high], [-high, -low, -low]])
        wall3 = torch.tensor([[-low, -low, -high], [high, low, low]])
        wall4 = torch.tensor([[-low, -low, -high], [-high, -low, -low]])
        walls_original = torch.cat([wall1, wall2, wall3, wall4], dim=1) # (2, 12)
        Rot_mat = self.rot_z(tau[:, -1]).cpu()  # (n, 2, 2)
        walls_rot = Rot_mat @ walls_original
        wall1_rot = walls_rot[:, :, :3]
        wall2_rot = walls_rot[:, :, 3:6]
        wall3_rot = walls_rot[:, :, 6:9]
        wall4_rot = walls_rot[:, :, 9:]
        return wall1_rot, wall2_rot, wall3_rot, wall4_rot
    
    def get_wrong_entrance(self, tau):
        low = 2.0
        # tau = (n, 3)
        ent1 = torch.tensor([[low, low], [-low, low]])
        ent2 = torch.tensor([[-low, low], [low, low]])
        ent3 = torch.tensor([[-low, -low], [-low, low]])
        ent4 = torch.tensor([[-low, low], [-low, -low]])
        ents_original = torch.cat([ent1, ent2, ent3, ent4], dim=1)
        Rot_mat = self.rot_z(tau[:, -1]).cpu()  # (n, 2, 2)
        ents_rot = Rot_mat @ ents_original
        ent1_rot = ents_rot[:, :, :2] # batch, 2
        ent2_rot = ents_rot[:, :, 2:4] 
        ent3_rot = ents_rot[:, :, 4:6]
        ent4_rot = ents_rot[:, :, 6:]
        
        # ent1_center = ent1_rot.mean(dim=-1)
        # ent2_center = ent2_rot.mean(dim=-1)
        # ent3_center = ent3_rot.mean(dim=-1)
        # ent4_center = ent4_rot.mean(dim=-1)
        
        init_points = tau[:, :2].cpu()
        init_points_no_rot = (Rot_mat.transpose(1, 2) @ init_points.unsqueeze(-1)).squeeze(-1)
        theta_init_points = torch.atan2(init_points_no_rot[:, 1], init_points_no_rot[:, 0])
        theta_1 = (theta_init_points >= 0) * (theta_init_points < np.pi/2)
        theta_2 = (theta_init_points >= np.pi/2) * (theta_init_points < np.pi)
        theta_3 = (theta_init_points >= -np.pi) * (theta_init_points < -np.pi/2)
        theta_4 = (theta_init_points >= -np.pi/2) * (theta_init_points < 0)
        # print(theta_1, theta_2, theta_3, theta_4)
        
        ent_selected_1 = torch.zeros_like(ent1_rot)
        ent_selected_2 = torch.zeros_like(ent1_rot)
        ent_selected_1[theta_1] = ent3_rot[theta_1]
        ent_selected_2[theta_1] = ent4_rot[theta_1]
        ent_selected_1[theta_2] = ent1_rot[theta_2]
        ent_selected_2[theta_2] = ent4_rot[theta_2]
        ent_selected_1[theta_3] = ent1_rot[theta_3]
        ent_selected_2[theta_3] = ent2_rot[theta_3]
        ent_selected_1[theta_4] = ent2_rot[theta_4]
        ent_selected_2[theta_4] = ent3_rot[theta_4]
        return ent_selected_1, ent_selected_2
        
    def _random_g(self, batch_size=10):
        random_rot = torch.randint(0, 4, (batch_size, 1))
        random_mirror = torch.randint(0, 2, (batch_size, 1))
        random_rot_wall = torch.rand((batch_size, 1)) * 0.5 * torch.pi - 0.25 * torch.pi
        random_g = torch.cat([random_rot, random_mirror, random_rot_wall], dim=1)
        return random_g
    
    
    def _random_task(self, batch_size=10):
        random_phi = torch.rand(batch_size, 1) * 2 * torch.pi
        random_r = torch.sqrt(torch.rand(batch_size, 1)) * 5 + 5
        random_x = torch.cos(random_phi) * random_r
        random_y = torch.sin(random_phi) * random_r
        random_theta = torch.rand((batch_size, 1)) * 0.5 * torch.pi - 0.25 * torch.pi
        random_tau = torch.cat([
            random_x, random_y, random_theta
        ], dim=1)
        return random_tau

    def _random_traj(self, batch_size=10):
        traj = (torch.rand(batch_size, 100, 2)*10)
        return traj
    
    def _random_data_aug(self, tau, traj=None, n_aug=1):
        n_data = len(tau)
        tau_aug_list = []
        traj_aug_list = []
        if traj is not None:
            original_shape = traj.shape
        if n_aug > 0:
            for i in range(n_aug):
                g = self._random_g(batch_size=n_data).to(tau)
                if traj is not None:
                    gtau, gtraj = self.action_traj(g, tau, traj)
                    traj_aug_list.append(gtraj)
                else:
                    gtau = self.action_task(g, tau)
                tau_aug_list.append(gtau)
            tau_aug = torch.cat(tau_aug_list, dim=0)
            if traj is not None:
                traj_aug = torch.cat(traj_aug_list, dim=0)
        else:
            tau_aug = tau
            traj_aug = traj
        if traj is not None:
            return tau_aug, traj_aug.reshape(-1, *original_shape[1:])
        else:
            return tau_aug
    
    # squeeze_hat_w and unsqueeze_hat_w : for reduced task parameter learning of equivariant_TCVAE
    def squeeze_hat_w(self, hat_w):
        return hat_w[:, :2]
    
    def unsqueeze_hat_w(self, squeezed_hat_w):
        batch_size = len(squeezed_hat_w)
        temp = torch.zeros((batch_size, 1)).to(squeezed_hat_w)
        return torch.cat((squeezed_hat_w, temp), dim=1)
    
    # extended_w and unextended_w : for extended task parameter learning of TCVAE
    def extend_w(self, w):
        if w.shape[-1] == 2:
            w = self.unsqueeze_hat_w(w)
        theta = w[:, 2]
        cos_theta = torch.cos(theta).unsqueeze(1)
        sin_theta = torch.sin(theta).unsqueeze(1)
        return torch.cat((w[:, :2], cos_theta, sin_theta), dim=1)
    
    def unextend_w(self, extended_w):
        theta = torch.atan2(extended_w[:, 3], extended_w[:, 2]).unsqueeze(1)
        return torch.cat((extended_w[:, :2], theta), dim=1)
    
    def project_group(self, tau):
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        g_rot_wall = -tau_theta
        xy_temp = (R_temp @ tau[:, :2].unsqueeze(-1)).squeeze(-1)
        
        batch_size = len(tau)
        g_rot = torch.zeros(batch_size).to(tau)
        g_mirror = torch.zeros(batch_size).to(tau)
        theta = torch.atan2(xy_temp[:, 1], xy_temp[:, 0])
        # r = torch.norm(xy_temp, dim=1)
        pi4 = torch.pi/4
        g_rot[(theta >= -pi4) * (theta < pi4)] = 0
        g_rot[(theta >= pi4) * (theta < 2 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 1
        g_rot[(theta >= 3* pi4) * (theta <= 4 * pi4) + (theta >= -4 * pi4) * (theta < -3 * pi4)] = 2
        g_rot[(theta >= 2* pi4) * (theta < 3 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 3
        
        g_mirror[(theta >= 0) * (theta < pi4) + (theta >= 2 * pi4) * (theta < 3 * pi4) + 
           (theta >= -4 * pi4) * (theta < -3 * pi4) + (theta >= -2 * pi4) * (theta < -pi4)] = 0
        g_mirror[(theta >= -pi4) * (theta < 0) + (theta >= pi4) * (theta < 2 * pi4) + 
           (theta >= 3 * pi4) * (theta <= 4 * pi4) + (theta >= -3 * pi4) * (theta < -2 * pi4)] = 1
        g = torch.cat([g_rot.unsqueeze(1), g_mirror.unsqueeze(1), g_rot_wall.unsqueeze(1)], dim=1)
        return g
        
    def get_inv(self, g):
        ginv = g.clone()
        ginv[g[:, 1] == 0, 0] =(-g[g[:, 1] == 0, 0]) % 4
        ginv[:, 2] = -ginv[:, 2]
        return ginv
    
    # def rot_matrix(self, theta):
    #     theta = torch.tensor(theta)
    #     c = torch.cos(theta)
    #     s = torch.sin(theta)
    #     R = torch.tensor([[c, -s], [s, c]])
    #     return
    
    def rot_z(self, theta):
        ct = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        R = torch.cat([
            torch.cat([ct, -st], dim=2),
            torch.cat([st, ct], dim=2)],
            dim=1)
        return R
    
    def action_task(self, g, tau):
        # mirroring first, rotation later
        self.R = self.R.to(tau)
        batch_size = len(g)
        g_rot = g[:, 0]
        g_mirror = g[:, 1]
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        xy_temp = (R_temp @ tau[:, :2].unsqueeze(-1)).squeeze(-1)
        
        # gtau = tau.clone()
        
        xy_temp[g_mirror == 1, 1] = -xy_temp[g_mirror == 1, 1]
        xy_temp[g_rot == 1] = torch.einsum('ij, kj -> ki', self.R[1], xy_temp[g_rot == 1])
        xy_temp[g_rot == 2] = torch.einsum('ij, kj -> ki', self.R[2], xy_temp[g_rot == 2])
        xy_temp[g_rot == 3] = torch.einsum('ij, kj -> ki', self.R[3], xy_temp[g_rot == 3])
        gtheta = (tau_theta + g[:, 2]) % (2*torch.pi)
        gtheta[gtheta > torch.pi] -= (2*torch.pi)
        R_temp2 = self.rot_z(gtheta)
        gxy = (R_temp2 @ xy_temp.unsqueeze(-1)).squeeze(-1)
        gtau = torch.cat([gxy, gtheta.unsqueeze(-1)], dim=1)
        # R_mul@gtau
        return gtau

    def action_traj(self, g, tau, traj):
        g = g.to(traj)
        tau = tau.to(traj)
        # traj: (bs, time_step, 2)
        batch_size = len(g)
        # time_step = traj.size()[1]
        traj = traj.reshape(batch_size, -1, 2)
        gtau = self.action_task(g, tau)
        
        gtraj = traj.clone()
        tau_theta = tau[:, -1]
        R_temp = self.rot_z(-tau_theta)
        gtraj = torch.einsum('nij, nkj -> nki', R_temp, gtraj)
        g_rot = g[:, 0]
        g_mirror = g[:, 1]
        gtraj[g_mirror == 1, :, -1] = -gtraj[g_mirror == 1, :, -1]
        gtraj[g_rot == 1] = torch.einsum('ij, klj -> kli', self.R[1], gtraj[g_rot == 1])
        gtraj[g_rot == 2] = torch.einsum('ij, klj -> kli', self.R[2], gtraj[g_rot == 2])
        gtraj[g_rot == 3] = torch.einsum('ij, klj -> kli', self.R[3], gtraj[g_rot == 3])
        gtheta = (tau_theta + g[:, 2]) % (2*torch.pi)
        gtheta[gtheta > torch.pi] -= (2*torch.pi)
        R_temp2 = self.rot_z(gtheta)
        gtraj = torch.einsum('nij, nkj -> nki', R_temp2, gtraj)
        return gtau, gtraj
    


# if __name__ == '__main__':
#     group1 = PouringGroup()
#     group2 = PlanarMobileRobot()
#     print("Pouring group test")
#     group1._check_valid()
#     print("2D Planar group test")
#     group2._check_valid()
