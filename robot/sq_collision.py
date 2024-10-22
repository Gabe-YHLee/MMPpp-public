from copy import deepcopy
import torch

def check_collision_base(
        Ts, 
        parameters,
        target_pc, 
        object_collision_scale=1.0,
    ):
    """_summary_

    Args:
        Ts (n_sq x 4 x 4 torch tensor): surrounding objects' poses
        parameters (n_sq x 5 torch tensor): surrounding objects' parameters
        target_pc (n_t x n_pc x 3): target's point cloud
        object_collision_scale (float, optional)

    Returns:
        scores (n_t bool torch tensor): scores for all targets
    """
 
    # bigger
    parameters_big = deepcopy(parameters)
    parameters_big[:, 0] *= object_collision_scale
    parameters_big[:, 1] *= object_collision_scale
    parameters_big[:, 2] *= object_collision_scale

    # flatten
    n_timesteps = target_pc.shape[0]
    n_pc = target_pc.shape[1]
    transformed_target_pc_flatten = target_pc.reshape(-1, 3)
    
    # calculate distance
    distances = sq_distance(
        transformed_target_pc_flatten, 
        Ts,
        parameters,
        mode='1'
    )
    
    distances = distances.reshape(n_timesteps, n_pc, -1)
    distances = torch.min(distances, dim=-1)[0]
    distances = torch.min(distances, dim=-1)[0]

    # calculate score
    scores = torch.zeros_like(distances)
    scores[distances > 1] = 0
    scores[distances <= 1] = 1

    return scores == 1

def check_collision(
        q_traj,
        Ts, 
        parameters,
        franka,
        object_collision_scale=1.0,
    ):
    """_summary_

    Args:
        q_traj (n_t x 7): joint angle trajectory
        Ts (n_sq x 4 x 4 torch tensor): surrounding objects' poses
        parameters (n_sq x 5 torch tensor): surrounding objects' parameters
        franka: deepcopied franka
        object_collision_scale (float, optional)

    Returns:
        scores (n_t bool torch tensor): scores for all targets
    """
    # bigger
    parameters_big = deepcopy(parameters)
    parameters_big[:, 0] *= object_collision_scale
    parameters_big[:, 1] *= object_collision_scale
    parameters_big[:, 2] *= object_collision_scale

    # match dimension
    pc_list = []
    for q in q_traj:
        franka.move_to_q(q)
        pc = franka.get_pcd_points(long=True)
        pc = torch.from_numpy(pc).float().unsqueeze(0) # n_pc x 3
        pc_list.append(pc)
    
    pc_list = torch.cat(pc_list, dim=0) # n_t x n_pc x 3

    # flatten
    n_t = pc_list.shape[0]
    n_pc = pc_list.shape[1]
    pc_list_flatten = pc_list.reshape(-1, 3)
    
    # calculate distance
    distances = sq_distance(
        pc_list_flatten, 
        Ts,
        parameters,
        mode='1'
    )
    
    distances = distances.reshape(n_t, n_pc, -1) # n_t x n_pc x n_sq
    distances = torch.min(distances, dim=-1)[0] # n_t x n_pc
    distances = torch.min(distances, dim=-1)[0] # n_t

    # calculate score
    scores = torch.zeros_like(distances)
    scores[distances > 1] = 0
    scores[distances <= 1] = 1
  
    return scores == 1

def sq_distance(x, poses, sq_params, mode='e1'):
    '''
    input: x : (n_pts x 3) pointcloud coordinates 
           poses : (n_sq x 4 x 4) superquadric poses
           sq_params : (n_sq x 5) superquadric parameters
           mode : 'e1' or '1'
    output: pointcloud sdf values for each superquadrics (n_pts x n_sq)
    '''    

    # parameters
    n_sq = len(sq_params)
    a1 = sq_params[:, [0]] 
    a2 = sq_params[:, [1]] 
    a3 = sq_params[:, [2]] 
    e1 = sq_params[:, [3]]
    e2 = sq_params[:, [4]]

    # object positions
    positions = poses[:, 0:3, 3] 
    rotations = poses[:, 0:3, 0:3] 

    # repeat voxel coordinates
    x = x.unsqueeze(0).repeat(n_sq, 1, 1).transpose(1,2) # (n_sq x 3 x n)

    # coordinate transformation
    rotations_t = rotations.permute(0,2,1)
    x_transformed = (
        - rotations_t @ positions.unsqueeze(2) 
        + rotations_t @ x
    ) # (n_sq x 3 x n) 

    # coordinates
    X = x_transformed[:, 0, :]
    Y = x_transformed[:, 1, :]
    Z = x_transformed[:, 2, :]

    # calculate beta
    F = (
        torch.abs(X/a1)**(2/e2)
        + torch.abs(Y/a2)**(2/e2)
        )**(e2/e1) + torch.abs(Z/a3)**(2/e1)

    if mode == 'e1':
        F = F ** e1

    return F.T