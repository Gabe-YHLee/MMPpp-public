import torch
from utils.LieGroup_torch_v2 import *
import time
import copy

from robot.robot_utils import squared_collision_distance

######################################
# A_screw: (n_joints, 6)             #  
# init link frames: (n_joints, 4, 4) #
# init eeframe: (4, 4)               #
# inertias: (n_joints, 6, 6)         #
######################################

def compute_S_screw(A_screw, initialLinkFrames_from_base):
    """_summary_
    Args:
        A_screw (torch.tensor): (n_joints, 6)
        initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
    """
    S_screw = []
    for M, A in zip(initialLinkFrames_from_base, A_screw):
        S_temp = Adjoint(M.unsqueeze(0)).squeeze(0)@A.unsqueeze(-1)
        S_screw.append(S_temp)
    S_screw = torch.cat(S_screw, dim=-1).permute(1, 0)
    return S_screw # (n_joints, 6)

def compute_B_screw(S_screw, initialEEFrame):
    B_screw = (Adjoint(inv_SE3(initialEEFrame)
        )@S_screw.transpose()).transpose()
    return B_screw # (n_joints, 6)

def forward_kinematics(jointPos, S_screw, initialLinkFrames_from_base, initialEEFrame, link_frames=True):
    """_summary_
    Args:
        jointPos (torch.tensor): (bs, n_joints)
        S_screw (torch.tensor): (n_joints, 6)
        initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
        M_01, M_02, ..., M_0n
        initialEEFrame (torch.tensor): (4, 4)
        M_0b
    """
    expSthetas = []
    n_joints = len(S_screw)
    bs = len(jointPos)
    temp = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(jointPos)
    for i in range(n_joints):
        temp = temp@exp_se3(
            jointPos[:, i].unsqueeze(1) * S_screw[i].unsqueeze(0))
        expSthetas.append(temp.unsqueeze(1))
    expSthetas = torch.cat(expSthetas, dim=1) # bs, n_links, 4, 4 
    EEFrame = expSthetas[:, -1]@initialEEFrame.unsqueeze(0) #bs, 4, 4
    if link_frames:
        LinkFrames_from_base = expSthetas @ initialLinkFrames_from_base.unsqueeze(0)
        return expSthetas, LinkFrames_from_base, EEFrame # (bs, n_joints, 4, 4), (bs, 4, 4)
    else: 
        return EEFrame # (bs, 4, 4)

def inverse_kinematics(
    initjointPos, 
    desiredEEFrame,
    robot,
    max_iter=5000, 
    step_size1=0.01, 
    step_size2=0.0001, 
    step_size3=0.0, 
    tolerance=0.001, 
    device='cpu',
    debug_mode=False
    ):
    """_summary_

    Args:
        initjointPos (torch.tensor): (bs, dof)
        desiredEEFrame (torch.tensor): (bs, 4, 4)
        robot (robot class): robot class
        max_iter (int, optional): Defaults to 5000.
        step_size1 (float, optional): Defaults to 0.01.
        step_size2 (float, optional): Joint limit. Defaults to 0.0001.
        step_size3 (float, optional): Self-collision. Defaults to 0.0.
        tolerance (float, optional): Defaults to 0.001.
    """
    bs, dof = initjointPos.size()
    jointPos = initjointPos
    iter_ = 0
    max_error = torch.inf
    JL_bool = False
    COL_bool = False
    while (iter_ < max_iter) and ((max_error > tolerance) or (not JL_bool) or (not COL_bool)):
        q = jointPos.clone().detach().requires_grad_(True)
        expSthetas, LinkFrames_from_base, EEFrame = forward_kinematics(
            q, 
            robot.S_screw,
            robot.initialLinkFrames_from_base,
            robot.initialEEFrame,
            )
        BodyJacobian = get_BodyJacobian(
            robot.S_screw, expSthetas, EEFrame) # (bs, 6, dof)
        
        dR = EEFrame[:, :3, :3].permute(0, 2, 1)@desiredEEFrame[:, :3, :3]
        wb = skew(log_SO3(dR)).view(-1, 3, 1) # (bs, 3, 1)
        vb = (EEFrame[:, :3, :3].permute(0, 2, 1)@(
            desiredEEFrame[:, :3, 3:] - EEFrame[:, :3, 3:])) # (bs, 3, 1)
        inv_Jb = approxmiate_pinv(BodyJacobian)
        jointVel = (inv_Jb@(torch.cat([wb, vb], dim=1))).view(bs, -1)
        jointPos += (step_size1*jointVel)
        
        null_proj = torch.eye(dof).view(1, dof, dof).to(
                    jointPos) - inv_Jb@BodyJacobian
        if step_size2 != 0:
            # joint limit
            q_limit = torch.tensor(robot.JointPos_Limits).to(device) # (2, 7)
            joint_limit_thr = copy.copy(q_limit)
            eps = 0.1
            joint_limit_thr[0, :] += eps 
            joint_limit_thr[1, :] -= eps
            
            low_violation_idx = joint_limit_thr[0:1, :] > jointPos
            high_violation_idx = joint_limit_thr[1:2, :] < jointPos
            
            Sigma = torch.zeros(bs, dof).to(jointPos)
            Sigma[low_violation_idx] = +(jointPos[low_violation_idx]-joint_limit_thr[0:1].repeat(bs, 1)[low_violation_idx])**2/eps**2
            Sigma[high_violation_idx] = -(jointPos[high_violation_idx]-joint_limit_thr[1:2].repeat(bs, 1)[high_violation_idx])**2/eps**2

            delta_joint = step_size2 * null_proj@Sigma.view(bs, dof, 1)
            jointPos += (delta_joint).view(bs, dof)

        if step_size3 !=0:
            min_sign_dist = squared_collision_distance(
                LinkFrames_from_base, 
                LinkFrames_from_base[:, -1]@robot.LLtoEE, 
                robot.link_capsules, 
                robot.gripper_capsules,
                show=False 
            )
            cost = (torch.relu(0.1 - min_sign_dist)**2).sum()
            if cost.item() > 0:
                cost.backward()
                delta_joint2 = step_size3 * null_proj@q.grad.view(bs, dof, 1)
                jointPos -= (delta_joint2).view(bs, dof)
                # if debug_mode:
                #     print(f"min_sign_dist: {min_sign_dist.max()}")
                    
        iter_ += 1
        error = torch.sum((EEFrame - desiredEEFrame).view(bs, -1)**2, dim=1)
        max_error = error.max()
        
        ## Joint limit check
        JointPos_Limits = torch.tensor(robot.JointPos_Limits).to(device)
        JL_bool = (jointPos - JointPos_Limits[0:1] > 0).prod().to(torch.bool) and (
            JointPos_Limits[1:2] - jointPos).prod().to(torch.bool).item()
        JL_bool = bool(JL_bool)
        
        ## Self-collision check
        COL_bool = (min_sign_dist > 0).prod().to(torch.bool).item()
        COL_bool = bool(COL_bool)
        
        if debug_mode:
            if iter_%100 == 0:
                print(
                    f'iter : {iter_}, max_error: {max_error}, joint limit check: {JL_bool}, self-collision check: {COL_bool}')
        
    if debug_mode:
        # Joint limit
        dict_infos = {
            'final_error': max_error,
            'joint limit check': JL_bool,
            "self-collision check": {COL_bool}
        }
        return jointPos, dict_infos
    else:
        return jointPos
       
def get_SpaceJacobian(S_screw, expSthetas):
    """_summary_
    Args:
        S_screw (torch.tensor): (n_joints, 6)
        expSthetas (torch.tensor): (bs, n_joints, 4, 4)
    """
    bs, n_joints, _, _ = expSthetas.size()
    SpaceJacobian = []
    SpaceJacobian.append(
        S_screw[0].unsqueeze(0).repeat(bs, 1).unsqueeze(-1) # (bs, 6, 1)
        )
    for i in range(n_joints-1):
        SpaceJacobian.append(
            (Adjoint(expSthetas[:, i])@S_screw[i+1].unsqueeze(0).unsqueeze(-1))
        )
    SpaceJacobian = torch.cat(SpaceJacobian, dim=-1)
    return SpaceJacobian # (bs, 6, n_joints)

def get_BodyJacobian(S_screw, expSthetas, EEFrame):
    """_summary_
    Args:
        S_screw (torch.tensor): (n_joints, 6)
        expSthetas (torch.tensor): (bs, n_joints, 4, 4)
        EEFrame (torch.tensor): (bs, 4, 4)
    """
    SpaceJacobian = get_SpaceJacobian(S_screw, expSthetas)
    BodyJacobian = Adjoint(inv_SE3(EEFrame))@SpaceJacobian
    return BodyJacobian # (bs, 6, n_joints)

def Large_L(LinkFrames_from_base):
    """_summary_
    Args:
        LinkFrames_from_base (torch.tensor): (bs, n_links, 4, 4)
    """
    bs, n_links, _, _ = LinkFrames_from_base.size()
    L = torch.eye(6*n_links).to(LinkFrames_from_base)
    L = L.unsqueeze(0).repeat(bs, 1, 1)
    
    for j in range(n_links):
        for i in range(n_links):
            if i > j:
                L[:, 6*i:6*(i+1), 6*j:6*(j+1)] = Adjoint(
                    inv_SE3(LinkFrames_from_base[:, i])@LinkFrames_from_base[:, j])
    return L # (bs, 6n, 6n)

def Large_A(A_screw):
    """_summary_
    Args:
        A_screw (torch.tensor): (n_joints, 6)
    """
    n = len(A_screw)
    A = torch.zeros(6*n, n).to(A_screw)
    for i in range(n):
        A[6*i:6*(i+1), i] = A_screw[i]
    return A # (6n, 6)

def Large_G(inertias):
    """_summary_
    Args:
        inertias (torch.tensor): (n_joints, 6, 6)
    """
    n_joints = len(inertias)
    G = torch.zeros(n_joints*6, n_joints*6).to(inertias)
    for i in range(n_joints):
        G[6*i:6*(i+1), 6*i:6*(i+1)] = inertias[i]
    return G

def Large_ad_V(V):
    """_summary_
    Args:
        V (torch.tensor): (bs, n_joints, 6)
    """
    bs, n_joints, _ = V.size()
    adV = adjoint(V.view(-1, 6)) # bs*n_joints, 6, 6
    Large_adV = torch.zeros(bs, 6*n_joints, 6*n_joints).to(V)
    for i in range(n_joints):
        Large_adV[:, 6*i:6*(i+1), 6*i:6*(i+1)] = adV.view(bs, n_joints, 6, 6)[:, i]
    return Large_adV # (bs, 6*n_joints, 6*n_joints)

def get_W(L):
    """_summary_
    Args:
        L (torch.tensor): (bs, 6*n_links, 6*n_links)
    """
    bs, six_n_linkes, _ = L.size()
    n_links = int(six_n_linkes/6)
    paddings = torch.zeros(bs, six_n_linkes, six_n_linkes).to(L)
    for i in range(n_links-1):
        paddings[:, 6*(i+1):6*(i+2), 6*i:6*(i+1)] = torch.ones_like(
            paddings[:, 6*(i+1):6*(i+2), 6*i:6*(i+1)]
            ).to(L)
    W = paddings*L
    return W
    
def get_MassMatrix(inertias, A_screw, LinkFrames_from_base):
    """_summary_
    Args:
        inertias (torch.tensor): (n_joints, 6, 6)
        A_screw (torch.tensor): (n_joints, 6)
        LinkFrames_from_base (torch.tensor): (bs, n_joints, 4, 4)
    """
    L = Large_L(LinkFrames_from_base)
    A = Large_A(A_screw)
    G = Large_G(inertias).unsqueeze(0)
    LA = L@A.unsqueeze(0)
    Mass = LA.permute(0, 2, 1)@G@LA
    return Mass
    
def get_Coriolis(inertias, A_screw, LinkFrames_from_base, jointVel):
    """_summary_
    Args:
        inertias (torch.tensor): (n_joints, 6, 6)
        A_screw (torch.tensor): (n_joints, 6)
        LinkFrames_from_base (torch.tensor): (bs, n_joints, 4, 4)
        jointVel: (torch.tensor): (bs, n_joints)
    """
    
    A = Large_A(A_screw).unsqueeze(0) # (1, 6n, 6)
    L = Large_L(LinkFrames_from_base) # (bs, 6n, 6n)
    G = Large_G(inertias).unsqueeze(0)  # (1, 6n, 6n)
    ad_Aqdot = Large_ad_V(A_screw.unsqueeze(0)*jointVel.unsqueeze(-1)) # (1, 6n, 6n)
    W = get_W(L) # (bs, 6n, 6n)
    V = (L@A@jointVel.unsqueeze(-1)).squeeze(-1)
    ad_V = Large_ad_V(V.view(-1, 7, 6)) # (bs, 6n, 6n)
    
    inner_term = G@L@ad_Aqdot@W + ad_V.permute(0, 2, 1)@G
    return (-A.permute(0, 2, 1)@L.permute(0, 2, 1)@inner_term@L@A@jointVel.unsqueeze(-1)).squeeze(-1)

def get_Gravity(inertias, A_screw, LinkFrames_from_base, base_acc=[0, 0, 0, 0, 0, 9.8]):
    """_summary_
    Args:
        inertias (torch.tensor): (n_joints, 6, 6)
        A_screw (torch.tensor): (n_joints, 6)
        LinkFrames_from_base (torch.tensor): (bs, n_joints, 4, 4)
    """
    bs, n_links, _, _ = LinkFrames_from_base.size()
    
    L = Large_L(LinkFrames_from_base)
    A = Large_A(A_screw)
    G = Large_G(inertias).unsqueeze(0)
    LA = L@A.unsqueeze(0)
    
    V_0 = torch.tensor([base_acc]).to(L) # 1, 6
    T_01 = LinkFrames_from_base[:, 0] # bs, 4, 4
    AdT_10 = Adjoint(inv_SE3(T_01)) # bs, 6, 6
    V_dot_base = (AdT_10@V_0.unsqueeze(-1)) # bs, 6, 1
    V_dot_base = torch.cat([V_dot_base, torch.zeros(bs, 6*(n_links-1), 1).to(L)], dim=1)
    return (LA.permute(0, 2, 1)@G@L@V_dot_base).squeeze(-1)

def get_Torque(qdot, qddot, inertias, A_screw, LinkFrames_from_base, base_acc=[0, 0, 0, 0, 0, 9.8]):
    """_summary_
    Args:
        inertias (torch.tensor): (n_joints, 6, 6)
        A_screw (torch.tensor): (n_joints, 6)
        LinkFrames_from_base (torch.tensor): (bs, n_joints, 4, 4)
        qdot: (torch.tensor): (bs, n_joints)
    """
    bs, n_links, _, _ = LinkFrames_from_base.size()
    
    A = Large_A(A_screw).unsqueeze(0) # (1, 6n, 6)
    L = Large_L(LinkFrames_from_base) # (bs, 6n, 6n)
    G = Large_G(inertias).unsqueeze(0)  # (1, 6n, 6n)
    ad_Aqdot = Large_ad_V(A_screw.unsqueeze(0)*qdot.unsqueeze(-1)) # (1, 6n, 6n)
    W = get_W(L) # (bs, 6n, 6n)
    V = (L@A@qdot.unsqueeze(-1)).squeeze(-1)
    ad_V = Large_ad_V(V.view(-1, 7, 6)) # (bs, 6n, 6n)
    
    LA = L@A
    Mass = LA.permute(0, 2, 1)@G@LA
    
    inner_term = G@L@ad_Aqdot@W + ad_V.permute(0, 2, 1)@G
    Coriolis =  (-A.permute(0, 2, 1)@L.permute(0, 2, 1)@inner_term@L@A@qdot.unsqueeze(-1)).squeeze(-1)
    
    V_0 = torch.tensor([base_acc]).to(L) # 1, 6
    T_01 = LinkFrames_from_base[:, 0] # bs, 4, 4
    AdT_10 = Adjoint(inv_SE3(T_01)) # bs, 6, 6
    V_dot_base = (AdT_10@V_0.unsqueeze(-1)) # bs, 6, 1
    V_dot_base = torch.cat([V_dot_base, torch.zeros(bs, 6*(n_links-1), 1).to(L)], dim=1)
    Gravity = (LA.permute(0, 2, 1)@G@L@V_dot_base).squeeze(-1)
    
    Torque = (Mass@qddot.unsqueeze(-1)).squeeze(-1) + Coriolis + Gravity
    return Torque

def get_LLVel_Torque_via_MatrixMul(
        q,
        qdot,
        qddot, 
        inertias, 
        A_screw, 
        S_screw, 
        initialLinkFrames_from_base, 
        base_acc=[0, 0, 0, 0, 0, 9.8]
    ):
        """_summary_
        Args:
            q: (torch.tensor): (bs, n_joints)
            qdot: (torch.tensor): (bs, n_joints)
            qddot: (torch.tensor): (bs, n_joints)
            inertias (torch.tensor): (n_joints, 6, 6)
            A_screw (torch.tensor): (n_joints, 6)
            S_screw (torch.tensor): (n_joints, 6)
            initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
        """
        bs, n_joints = q.size()
        expSthetas = []
        temp = torch.eye(4).view(1, 4, 4).repeat(bs, 1, 1).to(q)
        for i in range(n_joints):
            temp = temp@exp_se3(q[:, i:i+1] * S_screw[i:i+1])
            expSthetas.append(temp.view(bs, 1, 4, 4))
        expSthetas = torch.cat(expSthetas, dim=1) # bs, n_links, 4, 4 
        LinkFrames_from_base = expSthetas @ initialLinkFrames_from_base.view(1, n_joints, 4, 4)   
        
        A = Large_A(A_screw).view(1, 6*n_joints, n_joints) # (1, 6n, n)
        L = Large_L(LinkFrames_from_base) # (bs, 6n, 6n)
        G = Large_G(inertias).view(1, 6*n_joints, 6*n_joints)  # (1, 6n, 6n)
        ad_Aqdot = Large_ad_V(A_screw.view(1, n_joints, 6)*qdot.view(-1, n_joints, 1)) # (1, 6n, 6n)
        W = get_W(L) # (bs, 6n, 6n)
        V = (L@A@qdot.view(-1, n_joints, 1)).view(-1, 7, 6)
        ad_V = Large_ad_V(V) # (bs, 6n, 6n)
        
        LA = L@A
        Mass = LA.permute(0, 2, 1)@G@LA
        
        inner_term = G@L@ad_Aqdot@W + ad_V.permute(0, 2, 1)@G
        Coriolis =  -A.permute(0, 2, 1)@L.permute(0, 2, 1)@inner_term@L@A@qdot.view(-1, n_joints, 1)
        
        V_0 = torch.tensor([base_acc]).to(L) # 1, 6
        T_01 = LinkFrames_from_base[:, 0] # bs, 4, 4
        AdT_10 = Adjoint(inv_SE3(T_01)) # bs, 6, 6
        V_dot_base = (AdT_10@V_0.view(-1, 6, 1)) # bs, 6, 1
        V_dot_base = torch.cat([V_dot_base, torch.zeros(bs, 6*(n_joints-1), 1).to(L)], dim=1)
        Gravity = LA.permute(0, 2, 1)@G@L@V_dot_base
       
        LLVel = (LA@qdot.view(-1, 7, 1)).view(-1, 6*n_joints)[:, -6:]
        Torque = (Mass@qddot.view(-1, n_joints, 1) + Coriolis + Gravity).view(bs, n_joints)
        return LinkFrames_from_base, LLVel, Torque

def Newton_Euler_inverse_dynamics(
    q,
    qdot,
    qddot,
    A_screw,
    M,
    inertias,
    base_acc=[0, 0, 0, 0, 0, 9.8]
):
    """_summary_
    Args:
        q: (torch.tensor): (bs, n_joints)
        qdot: (torch.tensor): (bs, n_joints)
        qddot: (torch.tensor): (bs, n_joints)
        inertias (torch.tensor): (n_joints, 6, 6)
        A_screw (torch.tensor): (n_joints, 6)
        S_screw (torch.tensor): (n_joints, 6)
        M (torch.tensor): (n_joints, 4, 4)
        initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
        initialEEFrame (torch.tensor): (4, 4)
    """
    bs, n_joints = q.size()
    
    V = torch.zeros(bs, 6, 1).to(q) # (bs, 6, 1)
    V_dot = torch.tensor([base_acc]).to(q).repeat(bs, 1).view(-1, 6, 1) # (bs, 6, 1)
    
    V_all = torch.zeros(bs, n_joints, 6).to(q)
    V_dot_all = torch.zeros(bs, n_joints, 6).to(q)
    T_all = torch.zeros(bs, n_joints, 4, 4).to(q)
    
    # Forward iteration
    # tic = time.perf_counter()
    for i in range(n_joints):
        # tic1 = time.perf_counter()
        T = exp_se3(-skew(A_screw[i:i+1]*q[:, i:i+1]))@inv_SE3(M[i:i+1])
        # toc1 = time.perf_counter()
        # print(f"compute T: {toc1-tic1} s")
        
        # tic1 = time.perf_counter() 
        V = Adjoint(T)@V + (A_screw[i:i+1]*qdot[:, i:i+1]).view(-1, 6, 1)
        V_dot = Adjoint(T)@V_dot + \
            adjoint(V.squeeze(-1))@(A_screw[i:i+1]*qdot[:, i:i+1]).view(-1, 6, 1) + \
                (A_screw[i:i+1]*qddot[:, i:i+1]).view(-1, 6, 1)
        # toc1 = time.perf_counter() 
        # print(f"compute V, Vdot: {toc1-tic1} s")
        
        T_all[:, i, :, :] = T
        V_all[:, i, :] = V.view(-1, 6)
        V_dot_all[:, i, :] = V_dot.view(-1, 6)
    # toc = time.perf_counter() 
    # print(f"Forward Iter: {toc-tic} s")
        
    LLVel = V.view(-1, 6)
    
    # tic = time.perf_counter() 
    Torque = torch.zeros(bs, n_joints).to(q)
    for i in range(n_joints): 
        j = n_joints - i - 1
        F_j_term1 = inertias[j:j+1]@V_dot_all[:, j, :].view(-1, 6, 1)
        F_j_term2 = adjoint(V_all[:, j, :]).permute(0, 2, 1)@(inertias[j:j+1]@V_all[:, j, :].view(-1, 6, 1))
        if j == (n_joints - 1):
            F_j = F_j_term1 - F_j_term2 # (bs, n_joints, 1)
        else:
            F_j = Adjoint(T_all[:, j+1, :, :]).permute(0, 2, 1)@F_j + F_j_term1 - F_j_term2 # (bs, n_joints, 1)
        Torque[:, j] = (F_j.permute(0, 2, 1)@A_screw[j:j+1].view(-1, 6, 1)).view(-1)
    # toc = time.perf_counter() 
    # print(f"Backward Iter: {toc-tic} s")
    
    return LLVel, Torque