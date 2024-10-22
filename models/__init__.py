import os
from omegaconf import OmegaConf
import torch

from models.modules import (
    FC_vec,
    FC_SE32vec,
    FC_vec2SE3,
)

from models.models import (
    MMPpp,
    IMMPpp
)

from models.SE3_mmps import (
    discreteMMP,
    SE3MMPpp,
)

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] in [
            "fc_vec", 
            "fc_traj2latent",
            "fc_latent2traj",
            "fc_se32vec", 
            "fc_vec2se3", 
            "vf_fc_vec", 
            "vf_fc_se3"
        ]:
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        input_kwargs={
            'in_chan': in_dim,
            'out_chan': out_dim,
            'l_hidden': l_hidden,
            'activation': activation,
            'out_activation': out_activation
            }
        if kwargs["arch"] == "fc_vec":
            net = FC_vec(**input_kwargs)
        elif kwargs["arch"] == "fc_se32vec":
            net = FC_SE32vec(**input_kwargs)
        elif kwargs["arch"] == "fc_vec2se3":
            net = FC_vec2SE3(**input_kwargs)
    return net

def get_SE3_mmp(model_cfg, **kwargs):
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    
    b = model_cfg.get('b', 20)
    traj_len = model_cfg.get('traj_len', 480)
    iso_reg = model_cfg.get('iso_reg', 1)
    
    if arch == "discrete_SE3_mmp":
        encoder = get_net(in_dim=traj_len*12, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=traj_len*6, **model_cfg["decoder"])
        model = discreteMMP(encoder, decoder)
    elif arch == "se3mmppp":
        encoder = get_net(in_dim=(b+1)*6, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=(b+1)*6, **model_cfg["decoder"])
        model = SE3MMPpp(encoder, decoder, b=b)
    elif arch == "se3immppp":
        encoder = get_net(in_dim=(b+1)*6, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=(b+1)*6, **model_cfg["decoder"])
        model = SE3IMMPpp(encoder, decoder, b=b, iso_reg=iso_reg)
    return model

def get_mmp(model_cfg, **kwargs):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    
    dof = model_cfg.get('dof', 2)
    b = model_cfg.get('b', 20)
    via_points = model_cfg.get('via_points', [[0.8, 0.8], [-0.8, -0.8]])
    
    if arch == "mmppp":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = MMPpp(encoder, decoder, dof=dof, b=b, mode='vmp', via_points=via_points)
    elif arch == "immppp":
        metric = model_cfg.get("metric", "curve")
        iso_reg = model_cfg.get("iso_reg", 1.0)
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = IMMPpp(
            encoder, 
            decoder, 
            dof=dof, 
            b=b, 
            mode='vmp', 
            via_points=via_points,
            iso_reg=iso_reg,
            metric=metric,
            **kwargs)
    return model

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    model = _get_model_instance(name)
    model = model(model_dict, **kwargs)
    return model

def _get_model_instance(name):
    try:
        return {
            "mmppp": get_mmp,
            "immppp": get_mmp,
            "discrete_SE3_mmp": get_SE3_mmp,
            "se3mmppp": get_SE3_mmp,
            "se3immppp": get_SE3_mmp
        }[name]
    except:
        raise ("Model {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
   
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    
    model = get_model(cfg, **kwargs)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    if kwargs.get('eval', False):
        pretrained_dict = ckpt
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
    else:   
        model.load_state_dict(ckpt)
    
    return model, cfg