import numpy as np
from utils.LieGroup_numpy import *
from utils.utils import approximate_pinv

def get_BodyJacobianDeribative(q, q_dot, B_screw):
    n = len(q)
    dJ = np.zeros((6, n))
    Ad = np.eye(6)
    
    for i in [n-x-1 for x in range(n)]:
        dJ_i = np.zeros((6,n))
        T = np.eye(4)
        for j in [i-1-x for x in range(i)]:
            T = T@exp_se3(-B_screw[j+1,:]*q[j+1])
            dJ_i[:,j] = Adjoint_SE3(T)@B_screw[j,:]
        dJ = dJ + Ad@adjoint_se3(-B_screw[i,:])@dJ_i * q_dot[i]
        Ad = Ad@Adjoint_SE3(exp_se3(-B_screw[i,:] * q[i]))
    return dJ

def get_dBodyJacobian(q, B_screw):
    n = len(q)
    dJ = np.zeros((n, 6, n))
    Ad = np.eye(6)
    
    for i in [n-x-1 for x in range(n)]:
        dJ_i = np.zeros((6,n))
        T = np.eye(4)
        for j in [i-1-x for x in range(i)]:
            T = T@exp_se3(-B_screw[j+1,:]*q[j+1])
            dJ_i[:,j] = Adjoint_SE3(T)@B_screw[j,:]
        for k in range(n):
            dJ[k] = dJ[k] + Ad@adjoint_se3(-B_screw[i,:])@dJ_i * np.eye(n)[k,i]
        Ad = Ad@Adjoint_SE3(exp_se3(-B_screw[i,:] * q[i]))
    return dJ

def get_G(J_b, alpha=1):
    temp_ = J_b@J_b.transpose()
    temp = np.eye(J_b.shape[1]) - J_b.transpose()@approximate_pinv(temp_)@J_b
    return J_b.transpose()@J_b + alpha*temp

def get_dG(J_b, dJ_b, alpha=1):
    temp_ = J_b@J_b.transpose()
    J_bJ_bT_inv = approximate_pinv(temp_)
    I_minus_alpha_JbJBT_inv = np.eye(len(J_bJ_bT_inv)) - alpha * J_bJ_bT_inv
     
    term1 = dJ_b.transpose(0, 2, 1)@I_minus_alpha_JbJBT_inv@J_b # np.einsum('nij,jk,kl->nil')
    term2 = J_b.transpose()@I_minus_alpha_JbJBT_inv@dJ_b # np.einsum('ij,jk,nkl->nil')
    
    temp_term_left = J_b.transpose()@J_bJ_bT_inv
    temp_term_right = temp_term_left.transpose()
    
    term3 = alpha * temp_term_left@dJ_b@J_b.transpose()@temp_term_right
    term4 = alpha * temp_term_left@J_b@dJ_b.transpose(0, 2, 1)@temp_term_right
    
    return term1 + term2 + term3 + term4
    
def get_ChristoffelSymbol(G, dG):
    G_inv = approximate_pinv(G)
    Gamma = 0.5 * (
        np.einsum('ir, krj -> ijk', G_inv, dG) + \
        np.einsum('ir, jrk -> ijk', G_inv, dG) - \
        np.einsum('ir, rjk -> ijk', G_inv, dG) 
    ) 
    return Gamma

def potential_energy(EEFrame, T_g):
    R = EEFrame[:3, :3]
    p = EEFrame[:3, 3]
    Rg = T_g[:3, :3]
    pg = T_g[:3, 3]
    return 0.5*(log_SO3(R.transpose()@Rg)**2).sum() + ((p-pg)**2).sum()
    
def get_grad_potential_energy(EEFrame, J_b, T_g):
    J = []
    for x in J_b.transpose():
        J.append(skew(x).reshape(1, 4, 4))
    dT = EEFrame@np.concatenate(J, axis=0)
    dR = dT[:, :3, :3] # 7, 3, 3
    dp = dT[:, :3, 3]  # 7, 3
    
    R = EEFrame[:3, :3]
    p = EEFrame[:3, 3]
    Rg = T_g[:3, :3]
    pg = T_g[:3, 3]
    
    RTRg = R.transpose()@Rg
    p_minus_pg = p-pg
    
    v = np.zeros(7)
    for i in range(len(dR)):
        v[i] = np.trace(log_SO3(RTRg).transpose()@Dlog_SO3(RTRg, dR[i].transpose()@Rg))
        
    v += 2*dp@p_minus_pg.reshape(3)
    
    return v

    