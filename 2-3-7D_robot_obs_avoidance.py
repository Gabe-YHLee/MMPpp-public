import pybullet as p
import numpy as np
import time
import copy
from scipy.spatial.transform import Rotation as R
from assets.panda.panda import Panda
from envs.p2p import Point2PointTaskEnv
from utils.util_shelf import (
    box_urdf_writer,
    shelf_urdf_writer, 
    shelf_surface_points_sampler, 
    suface_points_sampler_box,
)
from models import load_pretrained
import torch
from lbf import Gaussian_basis, phi, vbf
from robot.openchains_lib import Franka
from robot.openchains_torch import forward_kinematics
from robot.capsule import batch_point2line_dist
from loader import get_dataloader
from utils.utils import fit_kde, sampling, log_kde

def check_collision_to_obs(q_future, robot, obs_position, rad=0.1, dist_out=False):
    expSthetas, LinkFrames_from_base, EEFrame = forward_kinematics(
        q_future.view(-1, 7), 
        robot.S_screw, 
        robot.initialLinkFrames_from_base, 
        robot.initialEEFrame, 
        link_frames=True
    )
    bs, n_dof, _, _ = LinkFrames_from_base.size()
    SE3poses = torch.cat(
        [
            torch.eye(4).view(1,1,4,4).repeat(bs, 1, 1, 1).to(LinkFrames_from_base),
            LinkFrames_from_base,
            EEFrame.view(-1, 1, 4, 4)
        ], dim=1
    ) # bs, n_dof+2, 4, 4
    capsules = torch.tensor(
        robot.link_capsules + robot.gripper_capsules
        ).to(LinkFrames_from_base) # ndof + 2, 7
    p1 = capsules[:, :3] # ndof + 2, 3
    p2 = capsules[:, 3:6] # ndof + 2, 3
    r = capsules[:, -1] # ndof + 2, 1
    Rp1 = torch.einsum('nmij, mj->nmi', SE3poses[:, :, :3, :3], p1)
    Rp2 = torch.einsum('nmij, mj->nmi', SE3poses[:, :, :3, :3], p2)
    Tp1 = Rp1 + SE3poses[:, :, :3, 3] # bs, ndof+2, 3
    Tp2 = Rp2 + SE3poses[:, :, :3, 3] # bs, ndof+2, 3

    ndof_dist_list = []
    for idof in range(n_dof+2):
        dist, _ = batch_point2line_dist(
            Tp1[:, idof, :], 
            Tp2[:, idof, :], 
            torch.tensor(
                obs_position, 
                dtype=torch.float32).view(1, 3).repeat(
                bs, 1)
        )
        ndof_dist_list.append(dist.unsqueeze(1))
    dist = torch.cat(ndof_dist_list, dim=1) # bs, 9
    if dist_out:
        return (dist.view(-1, 9) - r.view(1, 9) - rad).min(dim=1).values
    else:
        batch_min_dist = (dist.view(-1, 9) - r.view(1, 9) - rad).min(dim=1).values
        return (batch_min_dist > 0).prod().to(torch.bool)

# start PyBullet simulation 
enable_gui = True 
if enable_gui:
    p.connect(p.GUI)  # or p.DIRECT for non-graphical version
else:
    p.connect(p.DIRECT)  # non-graphical version
       
device = 'cpu'

# franka path
robot_path = 'assets/panda/panda_with_gripper.urdf'
env = Point2PointTaskEnv(
    robot_path=robot_path)
panda = Panda(T_ee=env.LastLink2EE)
robot = Franka(device=device)

############################################
############################################
############################################
########### GENERATE ENVIRONEMNT ###########
############################################
############################################
############################################

# gen shelf
shelf_path1 = "assets/shelf/shelf1.urdf" 
shelf_urdf_writer(
    shelf_path1, 0.4, 0.7, 0.7+0.4, np.array([0.1]), np.array([0.4+0.4]), 0.02)
shelf_surface_points1 = shelf_surface_points_sampler(
    4096, 0.4, 0.7, 0.7+0.4, np.array([0.1]), np.array([0.4+0.4]), 0.02)
shelf_path2 = "assets/shelf/shelf2.urdf" 
shelf_urdf_writer(
    shelf_path2, 0.4, 0.5, 0.8+0.4, np.array([]), np.array([0.25+0.3]), 0.02)
shelf_surface_points2 = shelf_surface_points_sampler(
    4096, 0.4, 0.5, 0.8+0.4, np.array([]), np.array([0.25+0.3]), 0.02)
shelf_table_path = "assets/shelf/table.urdf"
box_urdf_writer(
    shelf_table_path, 0.4, 0.4, 0.02
)
table_surface_points3 = suface_points_sampler_box(512, 0, 0, 0, 0.4, 0.4, 0.02)
shelf_table_path2 = "assets/shelf/table2.urdf"
box_urdf_writer(
    shelf_table_path2, 0.4, 0.4, 0.6
)
table_surface_points4 = suface_points_sampler_box(512, 0, 0, 0, 0.4, 0.4, 0.6)

# add shelf
shelf1_position = [0.7, 0.0, 0.0]
shelf1_orientation = [0.0, 0.0, 1., 0.]
shelf1_rot = np.asarray(
    p.getMatrixFromQuaternion(shelf1_orientation)).reshape(3, 3)

shelf_surface_points1 = (
    shelf1_rot@shelf_surface_points1.T + np.array(shelf1_position).reshape(3, 1)).T

shelf1_id = env.add_shelf(
    shelf_path1,
    shelf_position=shelf1_position,
    shelf_orientation=shelf1_orientation
)

shelf2_position = [0.0, -0.7, 0.0]
shelf2_orientation = [0.0, 0.0, 1., 0.]
shelf2_rot = np.asarray(
    p.getMatrixFromQuaternion(shelf2_orientation)).reshape(3, 3)@np.array([
            [0, 1, 0], 
            [-1, 0, 0], 
            [0, 0, 1]])
shelf2_orientation = (
    R.from_matrix(
        shelf2_rot
        )
    ).as_quat()

shelf_surface_points2 = (
    shelf2_rot@shelf_surface_points2.T + np.array(shelf2_position).reshape(3, 1)).T

shelf2_id = env.add_shelf(
    shelf_path2,
    shelf_position=shelf2_position,
    shelf_orientation=shelf2_orientation
)

shelf_table_position = [0.45, -0.3, 0.39+0.4]
shelf_table_orientation = [0.0, 0.0, 1., 0.]
shelf_table_rot = np.asarray(
    p.getMatrixFromQuaternion(shelf_table_orientation)).reshape(3, 3)

table_surface_points3 = table_surface_points3 + np.array([shelf_table_position])

shelf_table_orientation = (
    R.from_matrix(
        shelf_table_rot@np.array([
            [1, 0, 0], 
            [0, 1, 0], 
            [0, 0, 1]])
        )
    ).as_quat()
    
shelf_table_id = env.add_shelf(
    shelf_table_path,
    shelf_position=shelf_table_position,
    shelf_orientation=shelf_table_orientation
)

shelf_table2_position = [0.45, -0.7, 0.5]
table_surface_points4 = table_surface_points4 + np.array([shelf_table2_position])
shelf_table2_orientation = [0.0, 0.0, 1., 0.]
shelf_table2_rot = np.asarray(
    p.getMatrixFromQuaternion(shelf_table2_orientation)).reshape(3, 3)
shelf_table2_orientation = (
    R.from_matrix(
        shelf_table2_rot@np.array([
            [1, 0, 0], 
            [0, 1, 0], 
            [0, 0, 1]])
        )
    ).as_quat()
    
shelf_table2_id = env.add_shelf(
    shelf_table_path2,
    shelf_position=shelf_table2_position,
    shelf_orientation=shelf_table2_orientation
)

all_env_surface_points = np.concatenate(
    [
        shelf_surface_points1,
        shelf_surface_points2,
        table_surface_points3,
        table_surface_points4
    ], axis=0
)

visualize_surface_points_env = False
if visualize_surface_points_env:
    surface_points_id = []
    surface_points_visual_shape = []
    for pt in all_env_surface_points[np.random.permutation(len(all_env_surface_points))][:200]:
        surface_points_visual_shape.append(p.createVisualShape(
            shapeType=p.GEOM_SPHERE, 
            radius=0.02, rgbaColor=[0.5, 0.5, 0.5, 0.5]))
        surface_points_id.append(p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex=surface_points_visual_shape[-1],
            basePosition=pt))

init_ee_position = np.array([0.5, 0.15, 0.6+0.3])
init_ee_rot = np.array([
            [-1, 0, 0], 
            [0, -1, 0], 
            [0, 0, 1]])@np.array([
            [0, 0, -1], 
            [0, 1, 0], 
            [1, 0, 0]])

init_ee_pose = np.vstack([
        np.hstack([init_ee_rot, init_ee_position.reshape(3, 1)]), 
        np.array([0, 0, 0, 1])])

final_ee_position = np.array([0.0, -0.5, 0.65])
final_ee_rot = np.array([
            [0, 1, 0], 
            [-1, 0, 0], 
            [0, 0, 1]])@np.array([
            [-1, 0, 0], 
            [0, -1, 0], 
            [0, 0, 1]])@np.array([
            [0, 0, -1], 
            [0, 1, 0], 
            [1, 0, 0]])
            
init_ee_id = env.add_target_ee(
    'assets/panda/gripper_vis_only.urdf',
    basePosition=init_ee_position,
    baseOrientation=R.from_matrix(init_ee_rot).as_quat(),
    rgbaColor = [0.5, 0.5, 0.5, 0.5]
    )
final_ee_id = env.add_target_ee(
    'assets/panda/gripper_vis_only.urdf',
    basePosition=final_ee_position,
    baseOrientation=R.from_matrix(final_ee_rot).as_quat(),
    rgbaColor = [0.5, 0.5, 0.5, 0.5]
    )

    
q_i = [-1.22937435, -0.3774999, 1.17831243, -1.4427563, 0.28473847, 2.87936954, 0.85840966]
q_f = [-0.96280892, -0.2466983, -0.46373697, -2.23105282, 0.34272699, 3.62121556, 0.36830725]
 
restart_param = p.addUserDebugParameter("  restart", 1, 0, 0)  
demo_idx_param = p.addUserDebugParameter("  initial ref. traj.", 0, 9, 10)
obs_onoff_param = p.addUserDebugParameter("  obstacle (1: on, 0: off)", 0, 1, 2)   

rerun_past = p.readUserDebugParameter(restart_param)

#################### load model ####################

model, cfg = load_pretrained(
    root='results/robot-manifold',
    identifier='immppp_zdim2_reg1',
    config_file='immppp.yml',
    ckpt_file='model_best.pkl'
)
model.kwargs['via_points'] = [
    q_i,
    q_f
]

# Setup Dataloader
d_dataloaders = {}
for key, dataloader_cfg in cfg.data.items():
    d_dataloaders[key] = get_dataloader(dataloader_cfg)
ds = d_dataloaders['training'].dataset
idx = torch.sort(ds.targets.view(-1)).indices
ds.data = ds.data[idx]
ds.targets = ds.targets[idx]
w_dataset = model.get_w_from_traj(ds.data)
z_dataset = model.encode(w_dataset).detach()
zs_in_manifold = []
for z1, z2 in zip(z_dataset[:-1], z_dataset[1:]):
    for t in range(10):
        zs_in_manifold.append((1-t/9)*z1.view(1,-1) + z2.view(1,-1))
zs_in_manifold = torch.cat(zs_in_manifold, dim=0)     
local_cov, thr = fit_kde(z_dataset,  h_mul=0.5)

#####################################################

def generate_q_traj(time, w=None, model=None, mode='mmppp'):
    if mode == 'mmppp':
        basis_values = Gaussian_basis(time, b=model.b)
        return vbf(time, phi(basis_values), w, **model.kwargs)

def loss_func(z_samples, z_dataset, local_cov, thr, obs_position, q_future, robot, rad=0.1):
    logp = log_kde(z_samples, z_dataset, local_cov)
    prob_loss = torch.relu(torch.log(0.5*thr) - logp)**2
    dist = check_collision_to_obs(q_future, robot, obs_position, rad=rad, dist_out=True)
    col_loss = (torch.relu(0.005 - dist)**2).view(len(z_samples), -1).mean(dim=1)
    return prob_loss, col_loss

def optimize(current_z, model, obs_position, z, local_cov, thr, rad=0.1, timeState=0, mpc_time_interval=1, T=5, robot=robot, learning_rate=0.1):
    """_summary_

    Args:
        current_z (torch.tensor): (1, z_dim)
        model (torch.nn.Module): 
        obs_position (torch.tensor): (1, 3)
        z (torch.tensor): (n_basis, z_dim)
        local_cov (torch.tensor): (n_basis, z_dim, z_dim)
        thr (torch.tensor): (1, 1)
        rad (float, optional): _description_. Defaults to 0.1.
    """
    n_candidates = 0
    i_iter = 0
    flag = False
    while n_candidates == 0:
        n_samples = 100
        z_samples = sampling(n_samples, z, local_cov, thr, clipping=True)
        
        w = model.decode(z_samples).view(n_samples, model.b, model.dof)
        z_values = torch.tensor([timeState + i/9 * mpc_time_interval for i in range(10)]).view(1, -1, 1)/T
        z_values[z_values > 1] = 1
        q_future = generate_q_traj(z_values, w=w, model=model, mode='mmppp') # 100, 10, 7
        
        prob_loss, col_loss = loss_func(z_samples, z_dataset, local_cov, thr, obs_position, q_future, robot, rad=rad)
        loss = col_loss + prob_loss
        
        selected_z_samples = z_samples[loss < 1.0e-8]
        n_candidates = len(selected_z_samples)
        i_iter += 1
        if i_iter > 1:
            print(f"Fail to sample ... ")
            flag = False
            break
        flag = True
    
    if flag:
        dist_in_z = torch.norm(current_z - selected_z_samples, dim=1)
        min_idx = torch.argmin(dist_in_z)
        updated_z = z_samples[min_idx:min_idx + 1]
    else:
        updated_z = current_z
    return updated_z, flag

ts = 0.001
timeState = 0
p.setTimeStep(ts)
p.setRealTimeSimulation(1)
T = 5
env.reset_robot(jointPos=q_i)
replanning = False
_obstacle = None
mpc_time_interval = 0.5
dtau_range = 0.1
obs_onoff = 0
timeState = 0
init_time = time.time()
abs_init_time = time.time()

## INITIAL STATE ##
z_t = z_dataset[0:1].view(1, 2)
tau_t = torch.tensor([0.0]).view(1, 1, 1)
tic = time.time()
collision_free = True
q_prev_traj = copy.copy(torch.tensor(q_i).view(1, 7))
###################

while True:
    rerun = p.readUserDebugParameter(restart_param)
    
    if rerun != rerun_past:
        rerun_past = rerun
        demo_idx = int(p.readUserDebugParameter(demo_idx_param)+0.01)
        print(f"Demo idx reset: {demo_idx}")
        z_t = z_dataset[demo_idx:demo_idx+1]
        tau_t = torch.tensor([0.0]).view(1, 1, 1)
        w_t = model.decode(z_t).view(1, model.b, model.dof)
        
        obs_onoff = int(p.readUserDebugParameter(obs_onoff_param) > 0.5)
        if obs_onoff == 1:
            if _obstacle is not None:
                p.removeBody(_obstacle)
            s = torch.tensor(1/2).view(1, 1, 1)
            q_obs = generate_q_traj(s, w=w_t, model=model, mode='mmppp')
            obs_position = panda.solveForwardKinematics(
                q_obs.view(-1).detach().cpu().numpy()
                )[:3, 3]
            _obstacle = p.loadURDF(
                "assets/objects/sphere.urdf", 
                obs_position,
                useFixedBase=True)
            obs_position_center = copy.copy(obs_position)
            print(f"load obstacle at {obs_position}")
        
        elif obs_onoff == 0:
            if _obstacle is not None:
                p.removeBody(_obstacle)
            _obstacle = None

        env.reset_robot(jointPos=q_i)
        env.reset_gripper()
        
        init_time = time.time()
        abs_init_time = time.time()
        tic = time.time()
        
    ################################# POLICY RUN ###################################
    toc = time.time()
    timeState = toc - init_time
    if timeState > 1/10:
        infer_time_init = time.time()
        if obs_onoff == 1:
            z_values = tau_t + torch.tensor([i/9 * mpc_time_interval for i in range(10)]).view(1, -1, 1)/T
            z_values[z_values > 1] = 1
            q_future = generate_q_traj(z_values, w=w_t, model=model, mode='mmppp')
            collision_free = check_collision_to_obs(
                q_future, robot, obs_position, rad=0.1).item()
        else:
            collision_free = True

        if collision_free:
            tau_t = torch.min(tau_t + (toc - tic)/T, torch.ones_like(tau_t))
        else:
            # SAMPLING-BASED
            n_samples = 100
            z_samples = sampling(n_samples, z_dataset, local_cov, thr, clipping=True)
            dtau_samples = - dtau_range*torch.rand(n_samples).view(-1, 1, 1)
            tau_samples = tau_t + dtau_samples # bs, 1, 1
                  
            z_values = tau_samples + torch.tensor([i/9 * mpc_time_interval for i in range(10)]).view(1, -1, 1)/T # bs, 10, 1
            z_values[z_values > 1] = 1
            w_samples = model.decode(z_samples).view(-1, model.b, model.dof)
            q_traj_samples = generate_q_traj(z_values, w=w_samples, model=model, mode='mmppp') # 1000, 10, 7

            col_dist_out = check_collision_to_obs(
                q_traj_samples, 
                robot, 
                obs_position,
                rad=0.1, 
                dist_out=True
                ).view(-1)
            
            collision_free_idxs = col_dist_out.view(len(q_traj_samples), -1).min(dim=1).values > 0.0
            z_col_free_samples = z_samples[collision_free_idxs] # num_col_free, 2
            tau_col_free_samples = tau_samples[collision_free_idxs] # num_col_free, 1, 1
            
            z_connect = torch.cat([(1-s/9)*z_t.view(1, 1, 2) + s/9*z_col_free_samples.view(-1, 1, 2) for s in range(10)], dim=1) # bs, 10, 2
            log_KDE_vals = log_kde(z_connect.view(-1, 2), z_dataset, local_cov).view(-1, 10)
            indistidxs = torch.min(log_KDE_vals, dim=1).values > torch.log(0.1*thr)

            z_connect = z_connect[indistidxs]
            z_col_free_samples = z_col_free_samples[indistidxs]
            tau_col_free_samples = tau_col_free_samples[indistidxs]
            
            tau_connect = torch.cat([(1-s/9)*tau_t.view(1, 1, 1) + s/9*tau_col_free_samples.view(-1, 1, 1) for s in range(10)], dim=1) # bs, 10, 1
            w_samples = model.decode(z_connect.view(-1, 2)).view(-1, model.b, model.dof)
            q_connect_samples = generate_q_traj(tau_connect.view(-1, 1, 1), w=w_samples, model=model, mode='mmppp') # bsX10, 1, 7
            
            col_dist_out = check_collision_to_obs(
                q_connect_samples, 
                robot, 
                obs_position,
                rad=0.1, 
                dist_out=True
                ).view(-1)
            
            connect_col_free_idx = col_dist_out.view(-1, 10).min(dim=1).values > 0.0
            z_final_samples = z_col_free_samples[connect_col_free_idx]
            tau_final_samples = tau_col_free_samples[connect_col_free_idx]
            
            if len(z_final_samples) > 0:
                scores = torch.norm(z_t - z_final_samples, dim=1).view(-1) + 10000*torch.abs(tau_t - tau_final_samples).view(-1)
                min_idx = torch.argmin(scores)
                z_goal_t = z_final_samples[min_idx:min_idx+1]
                tau_goal_t = tau_final_samples[min_idx:min_idx+1]
            else:
                z_goal_t = z_t
                tau_goal_t = tau_t - 0.1
                
        init_time = time.time()
        print(f"Replanning time: {init_time - infer_time_init}")
    else:
        if collision_free:
            tau_t = torch.min(tau_t + (toc - tic)/T, torch.ones_like(tau_t))
        else:
            z_t = z_t + 100*(z_goal_t - z_t)*(toc-tic)
            tau_t = torch.min(tau_t + 100*(tau_goal_t - tau_t)*(toc-tic)/T, torch.ones_like(tau_t))
            
    tic = time.time()
    ################################################################################
    
    ## jittering obs COLLISION
    if obs_onoff == 1:
        current_time = time.time()
        obs_position = obs_position_center + np.array([0.1, 0.1, 0.0]) * np.sin(current_time-abs_init_time)
        p.resetBasePositionAndOrientation(
            _obstacle, 
            obs_position, 
            p.getQuaternionFromEuler([0,0,0]))
    
    ## Feedback POLICY: GENERATE q_traj from (z_t, w_t) ################
    w_t = model.decode(z_t).view(1, model.b, model.dof)
    q_traj = generate_q_traj(tau_t, w=w_t, model=model, mode='mmppp')
    ####################################################################
    
    if torch.norm(q_traj - q_prev_traj) > 0.001:
        q_traj = q_prev_traj + 0.001*(q_traj - q_prev_traj)/torch.norm(q_traj - q_prev_traj)
    
    env.move_joints(q_traj.view(-1).detach().cpu().numpy(), blocking=False)
    p.stepSimulation()
    q_prev_traj = copy.copy(q_traj)
