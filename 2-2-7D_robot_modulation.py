import os
import threading
import pybullet as p
import numpy as np
import time
import copy
from assets.panda.panda import Panda
import open3d as o3d

from lbf import Gaussian_basis, phi, vbf

import argparse

from models import load_pretrained

from loader import get_dataloader

import torch

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from datetime import datetime
import copy
from copy import deepcopy

from robot.franka import Franka

import urdfpy
from utils.utils import fit_kde, sampling

from envs.p2p import RobotEnv

# start PyBullet simulation 
enable_gui = False 
if enable_gui:
    p.connect(p.GUI)  # or p.DIRECT for non-graphical version
else:
    p.connect(p.DIRECT)  # non-graphical version
    
Q_i = [-1.22937435, -0.3774999, 1.17831243, -1.4427563, 0.28473847, 2.87936954, 0.85840966]
Q_f = [-0.96280892, -0.2466983, -0.46373697, -2.23105282, 0.34272699, 3.62121556, 0.36830725]
robot_path = 'assets/panda/panda_with_gripper.urdf'
env = RobotEnv(
    robot_path=robot_path)
panda = Panda(T_ee=env.LastLink2EE)
    
class AppWindow:
    def __init__(self):

        # argparser
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_root", default='datasets/robot-manifold')
        parser.add_argument("--pretrained_root", default='results/robot-manifold/')
        parser.add_argument("--identifier", default="immppp_zdim2_reg1")
        parser.add_argument("--config_file", default="immppp.yml")
        parser.add_argument("--ckpt_file", default="model_best.pkl")
        parser.add_argument("--device", type=str, default='any')
        args, unknown = parser.parse_known_args()

        # Setup device
        if args.device == "cpu":
            self.device = f"cpu"
        elif args.device == "any":
            self.device = f"cuda"
        else:
            self.device = f"cuda:{args.device}"

        # pretrained model
        self.immp, cfg = load_pretrained(
            args.identifier,
            args.config_file,
            args.ckpt_file,
            root=args.pretrained_root
        )
        self.immp.to(self.device)
        
        # Setup Dataloader
        d_dataloaders = {}
        for key, dataloader_cfg in cfg.data.items():
            d_dataloaders[key] = get_dataloader(dataloader_cfg)
        
        dataset_type = args.dataset_root.split('/')[-1]
        self.ds = d_dataloaders['training'].dataset
        
        ### fit GMM ###
        self.dataset_type = dataset_type
        if dataset_type == 'robot':
            self.immp.fit_GMM(
                d_dataloaders['training'].dataset.data.to(self.device), 
                n_components=2)
            self.n_samples_at_once = 100
            self.sample_idx = 0
            self.z_samples = self.sample_trajectory(mode='gmm')
        elif dataset_type == 'robot-manifold':
            idx = torch.sort(self.ds.targets.view(-1)).indices
            self.ds.data = self.ds.data[idx]
            self.ds.targets = self.ds.targets[idx]
            
            self.n_samples_at_once = 100
            self.sample_idx = 0
            
            w = self.immp.get_w_from_traj(self.ds.data.to(self.device))
            z = self.immp.encode(w)
            self.local_cov, self.thr = fit_kde(z, h_mul=0.5)
            self.z_samples = self.sample_trajectory(mode='kde', **{'z': z})
        ###############
        
        w_samples = self.immp.decode(self.z_samples).detach()
        w_samples = w_samples.view(self.n_samples_at_once, self.immp.b, self.immp.dof)
        z_values = torch.linspace(0, 1, 200).view(
                1, -1, 1).repeat(self.n_samples_at_once, 1, 1).to(self.device)
        basis_values = Gaussian_basis(
            z_values,
            b=self.immp.b)
        q_traj_samples = vbf(
            z_values, 
            phi(basis_values), 
            w_samples, 
            **self.immp.kwargs)
        self.q_traj = q_traj_samples[self.sample_idx].detach().cpu().numpy()
        
        ##
        self.z_values = torch.linspace(0, 1, 200).view(
                1, -1, 1).repeat(1, 1, 1).to(self.device)
        self.basis_values = Gaussian_basis(
            self.z_values,
            b=self.immp.b)
        self.w = w_samples[0:1]
        ##
        
        # ROBOT
        self.q_i = [-1.22937435, -0.3774999, 1.17831243, -1.4427563, 0.28473847, 2.87936954, 0.85840966]
        self.q_f = [-0.96280892, -0.2466983, -0.46373697, -2.23105282, 0.34272699, 3.62121556, 0.36830725]
        
        # Thread initialization
        self.event = threading.Event()
        self.thread_traj = threading.Thread(target=self.update_trajectory_video, daemon=True)
        # flag for visualization theread alignment
        self.flag_update_scene = None
        
        # parameters 
        image_size = [1024, 768]
        
        # Robot arm
        self.franka = Franka(
            azure=False, 
            root='', 
            add_bottle=False, 
            bottle_grasp_height=0.12)
        self.franka.gripper.gripper_width = 0.08
        self.franka.gripper.set_finger_width(0.065)
        self.vis_franka_idx = [0, 25, 50, 75, 100, 125, 150, 175, 199]

        # object material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        self.mat.base_color = [1.0, 1.0, 1.0, 0.9]
        self.mat_trans = rendering.MaterialRecord()
        self.mat_trans.shader = 'defaultLitTransparency'
        self.mat_trans.base_color = [1.0, 1.0, 1.0, 0.7]
        self.mat_env = rendering.MaterialRecord()
        self.mat_env.shader = 'defaultLit'
        self.mat_env.base_color = [.7686, .6431, .5176, 1.]
        
        self.vis_franka_mat = []
        for idx in self.vis_franka_idx:
            temp_mat = rendering.MaterialRecord()
            temp_mat.shader = 'defaultLitTransparency'
            task_progress = idx/199
            temp_mat.base_color = [
                0.3+(1-task_progress)*0.7, 
                0.3, 0.3+(task_progress)*0.7, 
                0.6]
            self.vis_franka_mat.append(temp_mat)
            
        mat_prev = rendering.MaterialRecord()
        mat_prev.shader = 'defaultLitTransparency'
        mat_prev.base_color = [1.0, 1.0, 1.0, 0.7]
        mat_coord = rendering.MaterialRecord()
        mat_coord.shader = 'defaultLitTransparency'
        mat_coord.base_color = [1.0, 1.0, 1.0, 0.87]

        # gen shelf  
        shelf_path1 = "assets/shelf/shelf1.urdf" 
        shelf_path2 = "assets/shelf/shelf2.urdf" 
        shelf_table_path = "assets/shelf/table.urdf"
        shelf_table_path2 = "assets/shelf/table2.urdf"
        os.makedirs("assets/shelf/shelf1/", exist_ok=True)
        os.makedirs("assets/shelf/shelf2/", exist_ok=True)
        os.makedirs("assets/shelf/table/", exist_ok=True)
        os.makedirs("assets/shelf/table2/", exist_ok=True)
        
        shelf_table_position = [0.45, -0.3, 0.39+0.4]
        shelf_table2_position = [0.45, -0.7, 0.5]

        shelf1_T = np.array([
            [-1, 0, 0, 0.7], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])
        shelf2_T = np.array([
            [0, -1, 0, 0], 
            [1, 0, 0, -0.7], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])
        
        self.list_mesh = []
        
        shelf1 = urdfpy.URDF.load(shelf_path1)
        shelf2 = urdfpy.URDF.load(shelf_path2)
        table = urdfpy.URDF.load(shelf_table_path)
        table2 = urdfpy.URDF.load(shelf_table_path2)
        for i, (key, val) in enumerate(shelf1.visual_trimesh_fk().items()):
            # e = key.export(file_type='obj')
            # with open(f"assets/shelf/shelf1/{i}.obj", "w") as f:
            #     f.write(e)
            temp = o3d.io.read_triangle_mesh(f"assets/shelf/shelf1/{i}.obj")
            temp.transform(val)
            temp.transform(shelf1_T)
            temp.compute_vertex_normals()
            self.list_mesh.append(
                copy.copy(temp)
            )
        for i, (key, val) in enumerate(shelf2.visual_trimesh_fk().items()):
            # e = key.export(file_type='obj')
            # with open(f"assets/shelf/shelf2/{i}.obj", "w") as f:
            #     f.write(e)
            temp = o3d.io.read_triangle_mesh(f"assets/shelf/shelf2/{i}.obj")
            temp.transform(val)
            temp.transform(shelf2_T)
            temp.compute_vertex_normals()
            self.list_mesh.append(
                copy.copy(temp)
            )
        for i, (key, val) in enumerate(table.visual_trimesh_fk().items()):
            # e = key.export(file_type='obj')
            # with open(f"assets/shelf/table/{i}.obj", "w") as f:
            #     f.write(e)
            temp = o3d.io.read_triangle_mesh(f"assets/shelf/table/{i}.obj")
            temp.transform(val)
            temp.translate(shelf_table_position)
            temp.compute_vertex_normals()
            self.list_mesh.append(
                copy.copy(temp)
            )
        for i, (key, val) in enumerate(table2.visual_trimesh_fk().items()):
            # e = key.export(file_type='obj')
            # with open(f"assets/shelf/table2/{i}.obj", "w") as f:
            #     f.write(e)
            temp = o3d.io.read_triangle_mesh(f"assets/shelf/table2/{i}.obj")
            temp.transform(val)
            temp.translate(shelf_table2_position)
            temp.compute_vertex_normals()
            self.list_mesh.append(
                copy.copy(temp)
            )
            
        ######################################################
        ################# STARTS FROM HERE ###################
        ######################################################

        # set window
        self.window = gui.Application.instance.create_window(
            str(datetime.now().strftime('%H%M%S')), 
            width=image_size[0], 
            height=image_size[1])
        w = self.window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # camera viewpoint
        self._scene.scene.camera.look_at(
            [0.8, -0.8, -0.5], # camera lookat
            [-0.45, 0.45, 1.7], # camera position
            [0, 0, 1] # fixed
        )

        # other settings
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.3, 0.3, -0.9))
        self._scene.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

        ############################################################
        ######################### MENU BAR #########################
        ############################################################
        
        # menu bar initialize
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # initialize collapsable vert
        inference_config = gui.CollapsableVert("Inference config", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        
        # bottle label angle        
        self._latent_coodrinates_slider = gui.Slider(gui.Slider.DOUBLE)
        self._latent_coodrinates_slider.set_limits(-1, 1)
        self._latent_coodrinates_slider.set_on_value_changed(self._set_latent_coordinates)
        inference_config.add_child(gui.Label("Latent coordinates"))
        inference_config.add_child(self._latent_coodrinates_slider)
        self.latent_value = 0 
        
        # add
        self._init_ee_pose_delta_y_slider = gui.Slider(gui.Slider.DOUBLE)
        self._init_ee_pose_delta_y_slider.set_limits(-0.1, 0.1)
        self._init_ee_pose_delta_y_slider.set_on_value_changed(self._set_ee_pose_y)
        
        inference_config.add_child(gui.Label("Start end-effector pose (y direction)"))
        inference_config.add_child(self._init_ee_pose_delta_y_slider)
        self.init_ee_pose_delta_y = 0
        
        # add
        self._final_ee_pose_delta_x_slider = gui.Slider(gui.Slider.DOUBLE)
        self._final_ee_pose_delta_x_slider.set_limits(-0.1, 0.1)
        self._final_ee_pose_delta_x_slider.set_on_value_changed(self._set_ee_pose_x)
        
        inference_config.add_child(gui.Label("Final end-effector pose (x direction)"))
        inference_config.add_child(self._final_ee_pose_delta_x_slider)
        self.final_ee_pose_delta_x = 0
        
        # Sample button
        self._sample_button = gui.Button("Perform random sampling")
        self._sample_button.horizontal_padding_em = 0.5
        self._sample_button.vertical_padding_em = 0
        self._sample_button.set_on_clicked(self._set_sample_mode)
        
        inference_config.add_fixed(separation_height)
        inference_config.add_child(self._sample_button)
        
        # Visualize type
        self._video_button = gui.Button("Video")
        self._video_button.horizontal_padding_em = 0.5
        self._video_button.vertical_padding_em = 0
        self._video_button.set_on_clicked(self._set_vis_mode_video)
        
        self._afterimage_button = gui.Button("Afterimage")
        self._afterimage_button.horizontal_padding_em = 0.5
        self._afterimage_button.vertical_padding_em = 0
        self._afterimage_button.set_on_clicked(self._set_vis_mode_afterimage)
        
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._video_button)
        h.add_child(self._afterimage_button)
        h.add_stretch()
        
        # add
        inference_config.add_child(gui.Label("Sample visualize type"))
        inference_config.add_child(h)

        # # direction
        # self._skip_size_silder = gui.Slider(gui.Slider.INT)
        # self._skip_size_silder.set_limits(5, 100)
        # self._skip_size_silder.set_on_value_changed(self._set_skip_size)
        # self.skip_size = 5
        
        # # add
        # inference_config.add_fixed(separation_height)
        # inference_config.add_child(gui.Label("Skip size"))
        # inference_config.add_child(self._skip_size_silder)
        
        self._settings_panel.add_child(inference_config)
        
        # add scene
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # # initial scene
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1)
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self.add_env()
        
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def end_thread(self, thread:threading.Thread):
        self.event.set()
        while thread.is_alive():
            time.sleep(0.1)
        self.event.clear()
    
    def reset_threads(self):
        if self.thread_traj.is_alive():
            self.end_thread(self.thread_traj)
        self.thread_traj = threading.Thread(
                            target=self.update_trajectory_video,
                            daemon=True)

    def add_env(self):
        # initial scene
        franka_ = deepcopy(self.franka)
        franka_.move_to_q(self.q_i)
        for i, mesh in enumerate(franka_.mesh):
            self._scene.scene.add_geometry(f'franka_mesh_{i}_init', mesh, self.vis_franka_mat[0])
        franka_.move_to_q(self.q_f)
        for i, mesh in enumerate(franka_.mesh):
            self._scene.scene.add_geometry(f'franka_mesh_{i}_final', mesh, self.vis_franka_mat[-1])
        
        for i, mesh in enumerate(self.list_mesh):
            self._scene.scene.add_geometry(
                f'env_{i}', 
                mesh, 
                self.mat_env)   

    def remove_trajectory(self):
        self._scene.scene.clear_geometry()
        self.add_env()
        
    def _set_vis_mode_video(self):
        self.reset_threads()
        self.thread_traj.start()

    def _set_vis_mode_afterimage(self):
        self.reset_threads()
        self.remove_trajectory()
        self.update_trajectory()

    def _set_sample_mode(self):
        if self.sample_idx > self.n_samples_at_once - 1:
            self.z_samples = self.sample_trajectory(n_samples=self.n_samples_at_once)
        self.z = self.z_samples[self.sample_idx:self.sample_idx+1]
        self.w = self.immp.decode(self.z).detach()
        self.w = self.w.view(1, self.immp.b, self.immp.dof)
        self.z_values = torch.linspace(0, 1, 201).view(
                1, -1, 1).repeat(1, 1, 1).to(self.device)
        self.basis_values = Gaussian_basis(
                self.z_values,
                b=self.immp.b) 
        self.sample_idx += 1
    
    def _set_latent_coordinates(self, value):
        self.latent_value = float(value)
        if self.dataset_type == 'robot':
            pass
        elif self.dataset_type == 'robot-manifold':
            ws = self.immp.get_w_from_traj(self.ds.data.to(self.device))
            zs = self.immp.encode(ws)
            
            idx = int(9*(self.latent_value + 1)/2)
            if idx == 9:
                selected_z = zs[idx]
            else:
                t = 9*(self.latent_value + 1)/2 - idx
                selected_z = (1-t)*zs[idx:idx+1] + t*zs[idx+1:idx+2]
                
            self.w = self.immp.decode(selected_z).view(1, self.immp.b, self.immp.dof)
            self.z_values = torch.linspace(0, 1, 200).view(
                1, -1, 1).repeat(1, 1, 1).to(self.device)
            self.basis_values = Gaussian_basis(
                self.z_values,
                b=self.immp.b)
        self._set_vis_mode_afterimage()
            
    def _set_ee_pose_x(self, value):
        self.final_ee_pose_delta_x = float(value)
        T_f = panda.solveForwardKinematics(Q_f)
        T_f[0, 3] += self.final_ee_pose_delta_x
        converge_flag, self.q_f = panda.solveInverseKinematics(
            T_f, 
            Q_f, 
            desired_q=Q_f,
            jointLowerLimits=np.array(env._robot_joint_lower_limit),
            jointUpperLimits=np.array(env._robot_joint_upper_limit)
            )
        if converge_flag:
            print("IK solved!")
            self._scene.scene.clear_geometry()
            self.add_env()
        else:
            print("IK does not converge!")
        self.immp.kwargs['via_points'] = [
            self.q_i,
            self.q_f
        ]
        self._set_vis_mode_afterimage()
            
    def _set_ee_pose_y(self, value):
        self.init_ee_pose_delta_y = float(value)
        T_i = panda.solveForwardKinematics(Q_i)
        T_i[1, 3] += self.init_ee_pose_delta_y
        converge_flag, self.q_i = panda.solveInverseKinematics(
            T_i, 
            Q_i, 
            desired_q=Q_i,
            jointLowerLimits=np.array(env._robot_joint_lower_limit),
            jointUpperLimits=np.array(env._robot_joint_upper_limit)
            )
        if converge_flag:
            print("IK solved!")
            self._scene.scene.clear_geometry()
            self.add_env()
        else:
            print("IK does not converge!")
        self.immp.kwargs['via_points'] = [
            self.q_i,
            self.q_f
        ]
        self._set_vis_mode_afterimage()
        
    def sample_trajectory(self, n_samples=100, traj_len=200, mode='gmm', **kwargs):
        if mode == 'gmm':
            dict_samples = self.immp.sample(
                n_samples, 
                device=self.device, 
                traj_len=traj_len, 
                clipping=True)
            rand_idx = torch.randperm(n_samples)
            z_samples = dict_samples['z_samples'][rand_idx]
            return z_samples 
        elif mode == 'kde':
            z = kwargs['z']
            z_samples = sampling(n_samples, z, self.local_cov, self.thr, clipping=True)
            return z_samples 
    
    def update_trajectory(self):
        self.q_traj = vbf(
            self.z_values,
            phi(self.basis_values), 
            self.w,
            **self.immp.kwargs
            )[0].detach().cpu().numpy()
        
        # update trajectory
        for mat, idx in zip(self.vis_franka_mat, self.vis_franka_idx):
            franka_ = deepcopy(self.franka)
            franka_.move_to_q(self.q_traj[idx])
            for i, mesh in enumerate(franka_.mesh):
                self._scene.scene.add_geometry(f'franka_mesh_{i}_{idx}', mesh, mat)
                            
    def update_trajectory_video(self):
        self.q_traj = vbf(
            self.z_values,
            phi(self.basis_values), 
            self.w,
            **self.immp.kwargs
            )[0].detach().cpu().numpy()
        
        self.flag_update_scene = [False] * len(self.q_traj)
        
        self.franka_ = deepcopy(self.franka)
        
        # update trajectory
        for idx in range(len(self.q_traj)):
            tic = time.time()
            if self.event.is_set():
                break
            self.idx = idx
            self.franka_.move_to_q(self.q_traj[idx])
            
            # Update geometry
            gui.Application.instance.post_to_main_thread(self.window, self.update_scene)
            # wait for dt
            dt = 0.03 
            while self.flag_update_scene[idx] == False:
                time.sleep(0.0001)
            toc = time.time()
            if toc - tic < dt:            
                time.sleep(dt - (toc - tic))
            else:
                print(f'idx = {idx}, time = {toc - tic}')

    def update_scene(self):
        self._scene.scene.clear_geometry()
        self.add_env()
        for i, mesh in enumerate(self.franka_.mesh):
            self._scene.scene.add_geometry(f'franka_mesh_{i}_{self.idx}', mesh, self.mat)
        self.flag_update_scene[self.idx] = True
        
if __name__ == "__main__":

    gui.Application.instance.initialize()

    w = AppWindow()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()