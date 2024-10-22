
import os
import open3d as o3d
import numpy as np
import sys
sys.path.append('..')
import torch
import utils.LieGroup_torch as lie

def get_mesh_bottle(
    path_bottle = os.path.join(
        'pouring_figures',
        'selected',
        'Bottle_selected',
        '1',
        'models',
        'model_normalized.obj'),
    bottle_center_z_offset=0.00,
    bottle_width=0.06,
    bottle_height=0.265,
    bottle_center_height=0.145
    ):
    mesh_bottle = o3d.io.read_triangle_mesh(path_bottle)
    R = mesh_bottle.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh_bottle.rotate(R, center=(0, 0, 0))
    mesh_bottle.translate([0, 0, bottle_center_z_offset])

    # bottle_rescalling_translation
    vertices_bottle_numpy = np.asarray(mesh_bottle.vertices) # n * 3 
    bottle_width_current = vertices_bottle_numpy[:, 0].max() - vertices_bottle_numpy[:, 0].min()
    bottle_height_current = vertices_bottle_numpy[:, 2].max() - vertices_bottle_numpy[:, 2].min()
    vertices_bottle_numpy[:, :2] *= bottle_width/bottle_width_current
    vertices_bottle_numpy[:, 2] *= bottle_height/bottle_height_current
    min_x_bottle = vertices_bottle_numpy[:, 0].min()
    min_z_bottle = vertices_bottle_numpy[:, 2].min()
    mesh_bottle.translate([(-bottle_width/2 - min_x_bottle), 0, -(bottle_center_height + min_z_bottle)])
    return mesh_bottle

def get_mesh_mug(
        path_mug=os.path.join(
        'pouring_figures',
        'selected',
        'Mug_selected',
        '1',
        'models',
        'model_normalized.obj'),
        mug_width_outer=0.08,
        mug_height=0.125                 
    ):
    mesh_mug = o3d.io.read_triangle_mesh(path_mug)
    R = mesh_mug.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    mesh_mug.rotate(R, center=(0, 0, 0))
    
    # mug_rescalling_translation
    vertices_mug_numpy = np.asarray(mesh_mug.vertices) # n * 3
    mug_width_current = vertices_mug_numpy[:, 0].max() - vertices_mug_numpy[:, 0].min()
    mug_height_current = vertices_mug_numpy[:, 2].max() - vertices_mug_numpy[:, 2].min()
    vertices_mug_numpy[:, :2] *= mug_width_outer/mug_width_current
    vertices_mug_numpy[:, 2] *= mug_height/mug_height_current
    min_y_mug = vertices_mug_numpy[:, 1].min()
    min_z_mug = vertices_mug_numpy[:, 2].min()
    mesh_mug.translate([0, (-mug_width_outer/2 - min_y_mug), -min_z_mug])
    return mesh_mug

def transform_demo(traj):
    traj_out = traj.clone()
    trans = torch.tensor([.89815, 0.20346 - 0.5, 0.22132+0.11])
    x = trans[0]
    y = trans[1]
    z = trans[2]
    traj_out[..., 0, -1] += x
    traj_out[..., 1, -1] += y
    traj_out[..., 2, -1] += z
    return traj_out

###########################################################################
###########################################################################
###########################################################################
###########################################################################
################################################### NOT MODIFIED YET BELOW:
def SE3_EE_to_bottle(traj):
    w2 = torch.zeros(1, 3)
    # w2[0, 2] = torch.pi/4
    w2[0, 2] = -3 * torch.pi/4
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee_to_bottle = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).detach()
    T_ee_to_bottle = T_ee_to_bottle.numpy()

    w = torch.zeros(1, 3)
    w[0, 1] = -torch.pi/2
    R = lie.exp_so3(w)
    p = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee = torch.cat([torch.cat([R, p], dim=2), eye], dim=1)
    T_ee = T_ee.numpy()
    traj_bottle = traj @ T_ee_to_bottle @ T_ee
    return traj_bottle

def SE3_bottle_to_EE(traj):
    w = torch.zeros(1, 3)
    alpha = 0
    w[0, 1] = torch.pi/2 + alpha
    R = lie.exp_so3(w)
    p = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee = torch.cat([torch.cat([R, p], dim=2), eye], dim=1).to(traj)
    
    w2 = torch.zeros(1, 3)
    w2[0, 2] = 3*torch.pi/4
    # w2[0, 2] = -torch.pi/4
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee2 = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).to(traj)
    traj_bottle = traj @ (T_ee @ T_ee2)
    return traj_bottle


def SE3_table_to_robot(traj):
    traj_out = traj.clone()
    trans = torch.tensor([.89815, 0.20346 - 0.5, 0.22132+0.11])
    x = trans[0]
    y = trans[1]
    z = trans[2]
    traj_out[..., 0, -1] += x
    traj_out[..., 1, -1] += y
    traj_out[..., 2, -1] += z
    w = torch.zeros(1, 3)
    alpha = 0
    w[0, 1] = torch.pi/2 + alpha
    R = lie.exp_so3(w)
    p = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee = torch.cat([torch.cat([R, p], dim=2), eye], dim=1).to(traj)
    
    w2 = torch.zeros(1, 3)
    w2[0, 2] = 3*torch.pi/4
    # w2[0, 2] = -torch.pi/4
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee2 = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).to(traj)
    traj_out = traj_out @ T_ee
    traj_out = traj_out @ T_ee2 
    return traj_out