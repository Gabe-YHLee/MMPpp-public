from stat import S_IEXEC
from xml.dom.pulldom import parseString
import numpy as np
from utils.LieGroup_numpy import *
from robot import Dynamics_numpy
import copy

def forward_kinematics(jointPos, S_screw, initialEEFrame):
    LinkFrames_from_base = []
    temp = np.eye(4)
    for q, S in zip(jointPos, S_screw):
        temp = temp@exp_se3(S*q)
        LinkFrames_from_base.append(temp.reshape(1,4,4))
    LinkFrames_from_base = np.concatenate(LinkFrames_from_base, axis=0)
    EEFrame = LinkFrames_from_base[-1]@initialEEFrame
    return LinkFrames_from_base, EEFrame

def get_SpaceJacobian(S_screw, LinkFrames_from_base):
    SpaceJacobian = []
    SpaceJacobian.append(S_screw[0].reshape(6,1))
    for T, S in zip(LinkFrames_from_base[:-1], S_screw[1:]):
        SpaceJacobian.append(
            (Adjoint_SE3(T)@S).reshape(6,1)
        )
    SpaceJacobian = np.concatenate(SpaceJacobian, axis=1)
    return SpaceJacobian

def get_BodyJacobian(S_screw, LinkFrames_from_base, EEFrame):
    SpaceJacobian = []
    SpaceJacobian.append(S_screw[0].reshape(6,1))
    for T, S in zip(LinkFrames_from_base[:-1], S_screw[1:]):
        SpaceJacobian.append(
            (Adjoint_SE3(T)@S).reshape(6,1)
        )
    SpaceJacobian = np.concatenate(SpaceJacobian, axis=1)
    BodyJacobian = Adjoint_SE3(
        inv_SE3(EEFrame)
    )@SpaceJacobian
    return BodyJacobian

def inverse_kinematics(
    initialEEFrame, 
    desiredEEFrame, 
    initjointPos, 
    S_screw, 
    B_screw=None, 
    max_iter=5000, 
    step_size=0.01, 
    step_size2=0.0001, 
    step_size3=0.0, 
    tolerance=0.001, 
    show=False, 
    joint_limit=None, 
    singular_avoidance=False, 
    debug=False,
    error_mode='geodesic'):
    jointPos = initjointPos
    iter = 0
    error = np.inf
    while (iter < max_iter) and (error > tolerance):
        LinkFrames_from_base, EEFrame = forward_kinematics(jointPos, S_screw, initialEEFrame)
        BodyJacobian = get_BodyJacobian(S_screw, LinkFrames_from_base, EEFrame)
        if error_mode == 'screw':
            V = pinv(BodyJacobian)@invskew( log_SE3( inv_SE3(EEFrame)@desiredEEFrame ) ).reshape(6,1)
        elif error_mode == 'geodesic':
            wb = invskew(log_SO3(EEFrame[:3, :3].transpose()@desiredEEFrame[:3, :3])).reshape(3,1)
            vb = EEFrame[:3, :3].transpose()@(desiredEEFrame[:3, 3:] - EEFrame[:3, 3:])
            V = pinv(BodyJacobian)@(np.concatenate([wb, vb], axis=0))
        jointPos += (step_size*V).reshape(-1)

        if (step_size2 != 0) and (step_size3 != 0):
            assert joint_limit is not None
            assert B_screw is not None
            assert singular_avoidance == True
            LinkFrames_from_base, EEFrame = forward_kinematics(jointPos, S_screw, initialEEFrame)
            J_b = get_BodyJacobian(S_screw, LinkFrames_from_base, EEFrame)
            G = Dynamics_numpy.get_G(J_b, alpha=0.1)
            to_center_joint_vec = joint_limit.mean(axis=1) - jointPos
            to_center_joint_vec = to_center_joint_vec/np.linalg.norm(to_center_joint_vec)
            delta_joint = step_size2 * (np.eye(7) - pinv(J_b)@J_b)@to_center_joint_vec.reshape(7, 1)
            jointPos += (delta_joint).reshape(-1)

            dJ_b = Dynamics_numpy.get_dBodyJacobian(jointPos.flatten(), B_screw)
            dG = Dynamics_numpy.get_dG(J_b, dJ_b, alpha=0.1)
            det_G = np.linalg.det(G)
            invG = np.linalg.pinv(G)
            singular_avoid =  np.einsum('nii -> n', np.expand_dims(det_G * invG, axis=0)@dG)
            singular_avoid = singular_avoid/np.linalg.norm(singular_avoid)
            singular_avoid = step_size3 * (np.eye(7) - pinv(J_b)@J_b)@singular_avoid.reshape(7, 1)
            jointPos += (singular_avoid).reshape(-1)
        elif (step_size2 != 0) and (step_size3 == 0):
            #################
            ## CURRENT USE ##
            #################
            assert joint_limit is not None
            LinkFrames_from_base, EEFrame = forward_kinematics(jointPos, S_screw, initialEEFrame)
            J_b = get_BodyJacobian(S_screw, LinkFrames_from_base, EEFrame)
            
            joint_limit_thr = copy.copy(joint_limit)
            eps = 0.1
            joint_limit_thr[:, 0] += eps 
            joint_limit_thr[:, 1] -= eps
            
            low_violation_idx = joint_limit_thr[:, 0] > jointPos
            high_violation_idx = joint_limit_thr[:, 1] < jointPos
            
            Sigma = np.zeros(7)
            Sigma[low_violation_idx] = +(jointPos[low_violation_idx]-joint_limit_thr[low_violation_idx, 0])**2/eps**2
            Sigma[high_violation_idx] = -(jointPos[high_violation_idx]-joint_limit_thr[high_violation_idx, 1])**2/eps**2
            
            delta_joint = step_size2 * (np.eye(7) - pinv(J_b)@J_b)@Sigma.reshape(7, 1)
            jointPos += (delta_joint).reshape(-1)
        elif (step_size2 == 0) and (step_size3 != 0):
            assert B_screw is not None
            assert singular_avoidance == True
            LinkFrames_from_base, EEFrame = forward_kinematics(jointPos, S_screw, initialEEFrame)
            J_b = get_BodyJacobian(S_screw, LinkFrames_from_base, EEFrame)
            dJ_b = Dynamics_numpy.get_dBodyJacobian(jointPos.flatten(), B_screw)
            dG = Dynamics_numpy.get_dG(J_b, dJ_b, alpha=0.1)
            det_G = np.linalg.det(G)
            invG = np.linalg.pinv(G)
            singular_avoid =  np.einsum('nii -> n', np.expand_dims(det_G * invG, axis=0)@dG)/det_G**3
            singular_avoid = singular_avoid/np.linalg.norm(singular_avoid)
            singular_avoid = step_size3 * (np.eye(7) - pinv(J_b)@J_b)@singular_avoid.reshape(7, 1)
            jointPos += (singular_avoid).reshape(-1)
        else:
            LinkFrames_from_base, EEFrame = forward_kinematics(jointPos, S_screw, initialEEFrame)
            J_b = get_BodyJacobian(S_screw, LinkFrames_from_base, EEFrame)
            
        iter += 1
        error = np.sum((EEFrame - desiredEEFrame)**2)
        if show:
            if iter%100:
                print(f'iter : {iter}, error: {error}')
    # print(f'iter : {iter}, error: {error}')
    if debug:
        detJJT = np.linalg.det(J_b@J_b.transpose())
        jointlimit_bool = np.prod((jointPos - joint_limit[:, 0]) > 0) * np.prod((jointPos - joint_limit[:, 1]) < 0)
        
        dict_infos = {
            'final_error': error,
            'detJJT': detJJT,
            'jointlimit_bool': jointlimit_bool
        }
        return jointPos, dict_infos
    else:
        return jointPos
