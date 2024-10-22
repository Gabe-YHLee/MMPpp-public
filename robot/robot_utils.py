import torch
from robot.capsule import (
    batch_line2line_dist
)
import utils.LieGroup_torch as lie

def squared_collision_distance(
    LinkFrames_from_base, # bs, n_dof, 4, 4
    EEFrame, # bs, 4, 4
    link_capsules, # n_dof+1, 7
    gripper_capsules, # 1, 7
    show=False,
):
    bs, n_dof, _, _ = LinkFrames_from_base.size()
    SE3poses = torch.cat(
        [
            torch.eye(4).view(1,1,4,4).repeat(bs, 1, 1, 1).to(LinkFrames_from_base),
            LinkFrames_from_base,
            EEFrame.view(-1, 1, 4, 4)
        ], dim=1
    ) 
    capsules = torch.tensor(
        link_capsules + gripper_capsules
        ).to(LinkFrames_from_base) # ndof + 2, 7
    p1 = capsules[:, :3] # ndof + 2, 3
    p2 = capsules[:, 3:6] # ndof + 2, 3
    r = capsules[:, -1] # ndof + 2, 1
    Rp1 = torch.einsum('nmij, mj->nmi', SE3poses[:, :, :3, :3], p1)
    Rp2 = torch.einsum('nmij, mj->nmi', SE3poses[:, :, :3, :3], p2)
    Tp1 = Rp1 + SE3poses[:, :, :3, 3] # bs, ndof+2, 3
    Tp2 = Rp2 + SE3poses[:, :, :3, 3] # bs, ndof+2, 3
    index_set = []
    P1 = []
    P2 = []
    Q1 = []
    Q2 = []
    R1 = []
    R2 = []
    for i in range(n_dof+2):
        for j in range(n_dof+2):
            if i>j:
                if (i-j > 2) and not ((i==8) and (j==5)):
                    index_set.append([i, j])
                    P1.append(Tp1[:, i, :])
                    P2.append(Tp2[:, i, :])
                    Q1.append(Tp1[:, j, :])
                    Q2.append(Tp2[:, j, :])
                    R1.append(r[i].view(-1))
                    R2.append(r[j].view(-1))
    P1 = torch.cat(P1) # (bs*len, 3)
    P2 = torch.cat(P2) # (bs*len, 3)
    Q1 = torch.cat(Q1) # (bs*len, 3)
    Q2 = torch.cat(Q2) # (bs*len, 3) 
    R1 = torch.cat(R1).view(1, -1) # (1, len)
    R2 = torch.cat(R2).view(1, -1) # (1, len)
    
    _, _, dist = batch_line2line_dist(P1, P2, Q1, Q2)
    
    sign_dist = dist.view(bs, -1) - R1 - R2
    
    min_sign_dist = sign_dist.min(dim=1).values
    if show:
        for i in range(len(min_sign_dist)):
            print(f"{i}: {min_sign_dist[i]}")
    return min_sign_dist
   
def traj_robotbase_EE_to_bottle_numpy_single(T):
    T_torch = torch.from_numpy(T).to(float).unsqueeze(0)
    T_torch_out = traj_robotbase_EE_to_bottle(T_torch)
    return T_torch_out.squeeze(0).numpy()

def traj_robotbase_EE_to_bottle(traj):
    w1 = torch.zeros(1, 3)
    w1[0, 2] = torch.pi/4
    R1 = lie.exp_so3(w1)
    p1 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee1 = torch.cat([torch.cat([R1, p1], dim=2), eye], dim=1).to(traj)
    T_bottle1 = torch.inverse(T_ee1)
    w2 = torch.zeros(1, 3)
    w2[0, 0] =  torch.pi/2
    R2 = lie.exp_so3(w2)
    p2 = torch.tensor([0.0, 0.0, 0.0]).reshape(1, 3, 1)
    eye = torch.eye(4)[-1].reshape(1, 1, 4)
    T_ee2 = torch.cat([torch.cat([R2, p2], dim=2), eye], dim=1).to(traj)
    T_bottle2 = torch.inverse(T_ee2)
    traj_out = traj @ T_bottle1 @ T_bottle2  
    return traj_out