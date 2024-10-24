import torch
import torch.nn as nn
import numpy as np

def relaxed_distortion_measure(
    func, 
    z, 
    eta=0.2, 
    metric='identity', 
    dim=2,
    H=None,
    *args, 
    **kwargs):
    
    bs = len(z)
    z_perm = z[torch.randperm(bs)]
    if eta is not None:
        alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha*z + (1-alpha)*z_perm
    else:
        z_augmented = z
    if metric == 'identity':
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=True)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()
        JTJv = (torch.autograd.functional.vjp(func, z_augmented, v=Jv, create_graph=True)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()
        return TrG2/torch.clip(TrG**2, min=1.0e-8)
    elif metric == 'curve':
        assert H is not None
        # H (1, -1, -1)
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(
            func, z_augmented, v=v, create_graph=True)[1].reshape(
                bs, -1, dim
            )
        TrG = torch.einsum(
            'bni, bnm, bmi -> b', Jv, H.to(z).repeat(bs, 1, 1), Jv
        ).mean()
        
        HJv = torch.einsum(
            'bnm, bmi -> bni', H.to(z).repeat(bs, 1, 1), Jv
        ).reshape(bs, -1)
        JTHJv = (torch.autograd.functional.vjp(
            func, z_augmented, v=HJv, create_graph=True)[1]).reshape(bs, -1)
        TrG2 = torch.sum(JTHJv**2, dim=1).mean()
        return TrG2/TrG**2
    else:
        raise NotImplementedError

def get_flattening_scores(G, mode='condition_number'):
    if mode == 'condition_number':
        S = torch.svd(G).S
        scores = S.max(1).values/S.min(1).values
    elif mode == 'variance':
        G_mean = torch.mean(G, dim=0, keepdim=True)
        A = torch.inverse(G_mean)@G
        scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
    else:
        pass
    return scores

def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

def get_pullbacked_Riemannian_metric(func, z):
    J = jacobian_decoder_jvp_parallel(func, z, v=None)
    G = torch.einsum('nij,nik->njk', J, J)
    return G
