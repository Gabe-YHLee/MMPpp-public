import torch
import torch.nn as nn

from vis_utils.plotly_SE3 import visualize_SE3
from utils.LieGroup_torch import log_SO3

from sklearn.mixture import GaussianMixture

from lbf import SE3LfD, Gaussian_basis, phi
from utils.LieGroup_torch import log_SO3, exp_so3, skew

from geometry import relaxed_distortion_measure
import math

class SE3MMPpp(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        b=30,
        h_mul=1,
        basis='Gaussian',
        **kwargs
        ):
        super(SE3MMPpp, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.b = b
        self.h_mul = h_mul
        self.basis = basis
        self.kwargs = kwargs
        
    def get_w_from_traj(self, x):
        '''
        x : (bs, L, 4, 4)
        '''
        w = SE3LfD(x, basis=self.basis, b=self.b, h_mul=self.h_mul, **self.kwargs) # (bs, b, dof)
        return w.view(len(x), -1).detach() # (bs, (b+1) * 6)

    def encode(self, w):
        return self.encoder(w) 
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, w):
        z = self.encode(w)
        recon = self.decode(z)
        return recon
    
    def compute_mse(self, x, recon):
        ## rotation part is weighted 10 times
        bs = len(x)
        xR = x.view(-1, 4, 4)[:, :3, :3]
        xp = x.view(-1, 4, 4)[:, :3, 3]
        reconR = recon.view(-1, 4, 4)[:, :3, :3] 
        reconp = recon.view(-1, 4, 4)[:, :3, 3] 
        mse_r = 0.5*(log_SO3(xR.permute(0,2,1)@reconR)**2).sum(dim=-1).sum(dim=-1)
        # mse_r = ((xR.permute(0,2,1)@reconR - torch.eye(3).unsqueeze(0).to(x))**2).sum(dim=-1).sum(dim=-1)
        mse_p = ((xp-reconp)**2).sum(dim=1)
        return 10*mse_r.view(bs, -1).mean(dim=1), mse_p.view(bs, -1).mean(dim=1)
        
        
    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        w = self.get_w_from_traj(x)
        recon = self(w)
        x_recon = self.w_to_SE3(recon, traj_len=x.size(1))
        mse_r, mse_p = self.compute_mse(x, x_recon)
        loss = mse_r.mean() + mse_p.mean()
        # loss = ((recon - w) ** 2).view(len(w), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, x, **kwargs):
        w = self.get_w_from_traj(x)
        recon = self(w)
        x_recon = self.w_to_SE3(recon, traj_len=x.size(1))
        mse_r, mse_p = self.compute_mse(x, x_recon)
        loss = mse_r.mean() + mse_p.mean()
        # loss = ((recon - w) ** 2).view(len(w), -1).mean(dim=1).mean()
        return {"loss": loss.item()}
    
    def fit_GMM(self, traj, n_components=2, **kwargs):
        w = self.get_w_from_traj(traj)
        z = self.encode(w).detach().cpu()
        self.gmm = GaussianMixture(
            n_components=n_components, 
            random_state=0,
            # reg_covar=0.1
            ).fit(z)
        self.gmm_thr = self.gmm.score_samples(z).min()
    
    def w_to_SE3(self, w, traj_len=100, tau_i=1.0e-5, tau_f=1-1.0e-5):
        n_samples = len(w)
        ws = w.view(n_samples, -1, 6)
        
        # tau = torch.linspace(tau_i, tau_f, traj_len).view(1, -1, 1).to(w) # (1, L, 1)
        tau = tau_i + (tau_f - tau_i)*torch.linspace(0, 1, traj_len).view(1, -1, 1).to(w)
        
        ## FIXED INIT POSE
        init_T = torch.tensor([
            [ 1,  0,  0,  0.6],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0.2],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).to(w)

        # final_T = torch.tensor([
        #         [ math.cos(2.3*3.141592/4),  0,  -math.sin(2.3*3.141592/4),  0.3],
        #         [ 0,  1,  0,  0],
        #         [ math.sin(2.3*3.141592/4),  0,  math.cos(2.3*3.141592/4),  0.23],
        #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).to(w)
        
        init_R = init_T[:3, :3].view(1, 3, 3)
        init_p = init_T[:3, 3].view(1, 3)
        # final_R = final_T[:3, :3].view(1, 3, 3)
        # final_p = final_T[:3, 3].view(1, 3)
        
        # init_logR = ws[:, 0, :3] # (bs, 3)
        # init_R = exp_so3(init_logR) # (bs, 3, 3)
        # init_p = ws[:, 0, 3:] # (bs, 3)
        
        final_logR = ws[:, 0, :3] # (bs, 3)
        final_R = exp_so3(final_logR) # (bs, 3, 3)
        final_p = ws[:, 0, 3:] # (bs, 3)
        
        w_R = ws[:, 1:, :3]
        w_p = ws[:, 1:, 3:] # (bs, b, 3)
        
        if self.basis == 'Gaussian':
            basis_values = Gaussian_basis(tau, b=self.b, h_mul=self.h_mul) # (1, L, b)
        else:
            raise NotImplementedError
        Phi = phi(basis_values).view(len(tau), traj_len, -1) # (1, L, b)

        multiplier = (1-tau)*tau
        p_samples = (1 - tau)*init_p.view(-1, 1, 3) + tau*final_p.view(-1, 1, 3) + multiplier*Phi@w_p # (bs, L, 3)

        log_nominalSO3 = log_SO3(
            init_R.permute(0, 2, 1)@final_R
        ).view(-1, 1, 3, 3)*tau.view(len(tau), -1, 1, 1)

        R_nominal = init_R.view(
            -1, 1, 3, 3)@exp_so3(
                log_nominalSO3.view(-1, 3, 3)).view(-1, traj_len, 3, 3)

        R_samples = R_nominal@exp_so3(
            (multiplier*Phi@w_R).view(-1, 3)
            ).view(n_samples, -1, 3, 3) # (bs, L, 3, 3)

        Rp_samples = torch.cat([
            R_samples, p_samples.view(n_samples, -1, 3, 1)
        ], dim=-1)
        zeros_ones = torch.tensor([0, 0, 0, 1.0], dtype=torch.float32).to(w).view(1, 1, 1, 4)

        x_samples = torch.cat([Rp_samples, zeros_ones.repeat(n_samples, traj_len, 1, 1)], dim=2)
        return x_samples
        
    def sample(self, n_samples, device=f'cpu', traj_len=480):
        dict_samples = {}
        num_data = 0
        list_z_samples = []
        list_cluster_samples = []
        gmm_thr = self.gmm_thr
        while num_data < n_samples:
            samples = self.gmm.sample(n_samples=n_samples)
            z_samples = torch.tensor(
                samples[0], dtype=torch.float32
                ).to(device)
            idx = self.gmm.score_samples(z_samples.detach().cpu()) > gmm_thr
            list_z_samples.append(z_samples[idx])
            list_cluster_samples.append(torch.tensor(samples[1])[idx])
            num_data = num_data + len(z_samples[idx])
        z_samples = torch.cat(list_z_samples, dim=0)[:n_samples]
        w_samples = self.decode(z_samples)
        
        dict_samples['x_samples'] = self.w_to_SE3(
            w_samples, traj_len=traj_len)
        dict_samples['z_samples'] = z_samples
        dict_samples['cluster_samples'] = torch.cat(
            list_cluster_samples, dim=0
        )[:n_samples]
        return dict_samples

class discreteMMP(nn.Module):
    def __init__(
            self, 
            encoder, 
            decoder, 
            smoothness_weight=10.0, 
            type_='SE3'
        ):
        super(discreteMMP, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.type = type_
        self.smoothness_weight = smoothness_weight
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def fit_GMM(self, traj, n_components=2, **kwargs):
        z = self.encode(traj).detach().cpu()
        self.gmm = GaussianMixture(
            n_components=n_components, 
            random_state=0
            ).fit(z)
        self.gmm_thr = self.gmm.score_samples(z).min()
    
    def sample(self, n_samples, device=f'cpu', traj_len=201):
        dict_samples = {}
        num_data = 0
        list_z_samples = []
        list_cluster_samples = []
        gmm_thr = self.gmm_thr
        while num_data < n_samples:
            samples = self.gmm.sample(n_samples=n_samples)
            z_samples = torch.tensor(
                samples[0], dtype=torch.float32
                ).to(device)
            idx = self.gmm.score_samples(z_samples.detach().cpu()) > gmm_thr
            list_z_samples.append(z_samples[idx])
            list_cluster_samples.append(torch.tensor(samples[1])[idx])
            num_data = num_data + len(z_samples[idx])
        z_samples = torch.cat(list_z_samples, dim=0)[:n_samples]
        dict_samples['x_samples'] = self.decode(z_samples)
        dict_samples['z_samples'] = z_samples
        dict_samples['cluster_samples'] = torch.cat(
            list_cluster_samples, dim=0)[:n_samples]
        return dict_samples
    
    def compute_mse(self, x, recon):
        if self.type == 'SE3':
            ## rotation part is weighted 10 times
            bs = len(x)
            xR = x.view(-1, 4, 4)[:, :3, :3]
            xp = x.view(-1, 4, 4)[:, :3, 3]
            reconR = recon.view(-1, 4, 4)[:, :3, :3] 
            reconp = recon.view(-1, 4, 4)[:, :3, 3] 
            mse_r = 0.5*(log_SO3(xR.permute(0,2,1)@reconR)**2).sum(dim=-1).sum(dim=-1)
            # mse_r = ((xR.permute(0,2,1)@reconR - torch.eye(3).unsqueeze(0).to(x))**2).sum(dim=-1).sum(dim=-1)
            mse_p = ((xp-reconp)**2).sum(dim=1)
            return 10*mse_r.view(bs, -1).mean(dim=1), mse_p.view(bs, -1).mean(dim=1)
        else:
            return torch.zeros(1), ((x-recon)**2).mean(dim=-1)
        
    def smoothness_loss(self, z, eta=0.4):
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z
        x = self.decode(z_augmented)
        if self.type == 'SE3':
            xdot = x[:, 1:, :, :] - x[:, :-1, :, :]
            xdot = xdot.view(len(x), -1, 16)
            energy = ((xdot)**2).sum(dim=-1).mean(dim=-1)
        else:
            bs, traj_len, x_dim = x.size()
            xdot = x[:, 1:] - x[:, :-1]
            xdot = xdot.view(bs, -1, x_dim)
            energy = ((xdot)**2).sum(dim=-1).mean(dim=-1)
        return energy
     
    def train_step(self, x, optimizer=None, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        mse_r, mse_p = self.compute_mse(x, recon)
        loss = mse_r.mean() + mse_p.mean()
        
        z = self.encode(x)
        energy = self.smoothness_loss(z)
        loss = loss + self.smoothness_weight * energy.mean()
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "mse_r_": mse_r.mean().item(), "mse_p_": mse_p.mean().item(), "energy_": energy.mean().item()}
    
    def validation_step(self, x, **kwargs):
        recon = self(x)
        mse_r, mse_p = self.compute_mse(x, recon)
        loss = mse_r.mean() + mse_p.mean()
        return {"loss": loss.item(), "rmse_r_": torch.sqrt(mse_r.mean()).item(), "rmse_p_": torch.sqrt(mse_p.mean()).item()}

    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]
        for data in dl:
            x = data[0]
            break
        skip_size = 5
        recon = self(x.to(device))
        fig = visualize_SE3(
            x.detach().cpu()[:, 0: -1: skip_size], 
            recon.detach().cpu()[:, 0: -1: skip_size]
        )
        return {"SE3traj_data_and_recon#": fig}