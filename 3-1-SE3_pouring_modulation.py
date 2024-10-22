import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import argparse
from datetime import datetime
from copy import deepcopy

from vis_utils.open3d_utils import (
    get_mesh_bottle, 
    get_mesh_mug, 
)
from models import load_pretrained
from loader import get_dataloader

import threading
import time

from utils.LieGroup_torch import inverse_SE3
from utils.utils import SE3smoothing

class AppWindow:
    def __init__(self):
        # argparser
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_root", default='datasets/bigbottle_pkl_subdataset/pouring_data')
        parser.add_argument("--skip_size", type=int, default=3)
        parser.add_argument("--pretrained_root", default='results/SE3')
        parser.add_argument("--identifier", default="se3mmppp/mmppp_zdim2")
        parser.add_argument("--method", default="mmppp")
        parser.add_argument("--config_file", default="mmppp.yml")
        parser.add_argument("--ckpt_file", default="model_best.pkl")
        parser.add_argument("--device", type=str, default='any')
        parser.add_argument("--se3_traj_smoothing", type=bool, default=False)
        args, unknown = parser.parse_known_args()

        self.method = args.method
        
        # Setup device
        if args.device == "cpu":
            self.device = f"cpu"
        elif args.device == "any":
            self.device = f"cuda"
        else:
            self.device = f"cuda:{args.device}"

        self.mmp, cfg = load_pretrained(
            args.identifier,
            args.config_file,
            args.ckpt_file,
            root=args.pretrained_root
        )
        self.mmp.to(self.device)
        
        # Setup Dataloader
        d_dataloaders = {}
        for key, dataloader_cfg in cfg.data.items():
            d_dataloaders[key] = get_dataloader(dataloader_cfg)
        self.ds = d_dataloaders['training'].dataset
        
        ######## Fit GMM ########
        self.mmp.fit_GMM(self.ds.traj_data.to(self.device), n_components=2)
        dict_samples = self.mmp.sample(100, device=self.device, traj_len=480)
        x_samples = dict_samples['x_samples']
        self.trajs = x_samples.detach().cpu()
        # if args.se3_traj_smoothing:
        #     self.trajs = SE3smoothing(self.trajs)
        ########################
        
        # Thread initialization
        self.event = threading.Event()
        self.thread_traj = threading.Thread(
            target=self.update_trajectory_video, daemon=True)
        # flag for visualization theread alignment
        self.flag_update_scene = None
        
        self.skip_size = args.skip_size
        
        # parameters 
        image_size = [1024, 768]
        
        # mesh table
        self.mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=2, height=2, depth=0.03
        )
        self.mesh_box.translate([-1, -1, -0.03])
        self.mesh_box.paint_uniform_color([222/255,184/255,135/255])
        self.mesh_box.compute_vertex_normals()

        # Obstacle 
        self.mesh_obs = o3d.geometry.TriangleMesh.create_box(
            width=0.1, height=0.1, depth=0.3
        )
        self.mesh_obs.translate([0.4, -0.05, 0.0])
        self.mesh_obs.paint_uniform_color([222/255, 50/255, 50/255])
        self.mesh_obs.compute_vertex_normals()
        
        # bottle label
        self.draw_bottle_label = True
        self.bottle_label_angle = 0

        # frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1
        )

        # object material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        self.mat.base_color = [1.0, 1.0, 1.0, 0.9]
        mat_prev = rendering.MaterialRecord()
        mat_prev.shader = 'defaultLitTransparency'
        mat_prev.base_color = [1.0, 1.0, 1.0, 0.7]
        mat_coord = rendering.MaterialRecord()
        mat_coord.shader = 'defaultLitTransparency'
        mat_coord.base_color = [1.0, 1.0, 1.0, 0.87]

        ######################################################
        ################# STARTS FROM HERE ###################
        ######################################################

        # set window
        self.window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
        w = self.window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # camera viewpoint
        self._scene.scene.camera.look_at(
            [0, 0, 0], # camera lookat
            [0.7, 0, 0.9], # camera position
            [0, 0, 1] # fixed
        )

        # other settings
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.3, 0.3, -0.9))
        self._scene.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

        # mesh list
        self.bottle_idx = 1
        self.mug_idx = 4

        # load bottle
        self.mesh_bottle = get_mesh_bottle(
            root='./3dmodels/bottles', 
            bottle_idx=self.bottle_idx
        )
        self.mesh_bottle.compute_vertex_normals()

        # color template (2023 pantone top 10 colors)
        rgb = np.zeros((10, 3))
        rgb[0, :] = [208, 28, 31] # fiery red
        rgb[1, :] = [207, 45, 113] # beetroot purple
        rgb[2, :] = [249, 77, 0] # tangelo
        rgb[3, :] = [250, 154, 133] # peach pink
        rgb[4, :] = [247, 208, 0] # empire yellow
        rgb[5, :] = [253, 195, 198] # crystal rose
        rgb[6, :] = [57, 168, 69] # classic green
        rgb[7, :] = [193, 219, 60] # love bird 
        rgb[8, :] = [75, 129, 191] # blue perennial 
        rgb[9, :] = [161, 195, 218] # summer song
        rgb = rgb / 255  

        # new bottle coloring
        bottle_vertices = np.asarray(self.mesh_bottle.vertices) 
        bottle_normals = np.asarray(self.mesh_bottle.vertex_normals)
        bottle_colors = np.ones_like(bottle_normals)
        bottle_colors[:, :3] = rgb[8]
        # z_values = bottle_vertices[:, 2]
        # print(np.max(z_values), np.min(z_values))
        # bottle_colors[np.logical_and(z_values > 0.17, z_values < 0.2)] = rgb[6]
        self.mesh_bottle.vertex_colors = o3d.utility.Vector3dVector(
            bottle_colors
        )

        if self.draw_bottle_label:
            self.mesh_bottle_label = o3d.geometry.TriangleMesh.create_cylinder(radius=0.0355, height=0.07, resolution=30, create_uv_map=True, split=20)
            self.mesh_bottle_label.paint_uniform_color([0.8, 0.8, 0.8])
            self.mesh_bottle_label.compute_vertex_normals()

            # initialize
            bottle_cylinder_vertices = np.asarray(self.mesh_bottle_label.vertices)
            bottle_cylinder_normals = np.asarray(self.mesh_bottle_label.vertex_normals)
            bottle_cylinder_colors = np.ones_like(bottle_cylinder_normals)
            # print(np.min(bottle_cylinder_vertices), np.max(bottle_cylinder_vertices))
            
            # band
            n = bottle_cylinder_normals[:, :2]
            n = n/np.linalg.norm(n, axis=1, keepdims=True)
            bottle_cylinder_colors[
                np.logical_and(np.logical_and(n[:, 0] > 0.85, bottle_cylinder_vertices[:, 2] > -0.025), bottle_cylinder_vertices[:, 2] < 0.025)
            ] = rgb[0]
            
            self.mesh_bottle_label.vertex_colors = o3d.utility.Vector3dVector(
                bottle_cylinder_colors
            )

            self.mesh_bottle_label.translate([0.0, 0, 0.155])
            R = self.mesh_bottle_label.get_rotation_matrix_from_xyz((0, 0, self.bottle_label_angle))
            self.mesh_bottle_label.rotate(R, center=(0, 0, 0))

        # # # combine
        # self.mesh_bottle += self.mesh_bottle_label

        # load mug
        self.mesh_mug = get_mesh_mug(
            root='./3dmodels/mugs', 
            mug_idx=self.mug_idx
        )
        self.mesh_mug.paint_uniform_color(rgb[2] * 0.6)
        self.mesh_mug.compute_vertex_normals()
        
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


        # self._samples_slider = gui.Slider(gui.Slider.DOUBLE)
        # self._samples_slider.set_limits(0, len(self.trajs))
        # self._samples_slider.set_on_value_changed(self._set_sample_idx)
        # inference_config.add_child(gui.Label("Samples idx"))
        # inference_config.add_child(self._samples_slider)
        # self.traj_idx = 0 
        
        self._latent_coodrinates_slider = gui.Slider(gui.Slider.DOUBLE)
        self._latent_coodrinates_slider.set_limits(-1, 1)
        self._latent_coodrinates_slider.set_on_value_changed(
            self._set_latent_coordinates)
        inference_config.add_child(gui.Label("Latent coordinates"))
        inference_config.add_child(self._latent_coodrinates_slider)
        self.latent_value = 0 
        
        self._cup_x_slider = gui.Slider(gui.Slider.DOUBLE)
        self._cup_x_slider.set_limits(-0.2, 0.2)
        self._cup_x_slider.set_on_value_changed(
            self._set_cup_x)
        inference_config.add_child(gui.Label("Cup X coordinates"))
        inference_config.add_child(self._cup_x_slider)
        self.cup_x = 0 
        
        self._cup_y_slider = gui.Slider(gui.Slider.DOUBLE)
        self._cup_y_slider.set_limits(-0.2, 0.2)
        self._cup_y_slider.set_on_value_changed(
            self._set_cup_y)
        inference_config.add_child(gui.Label("Cup Y coordinates"))
        inference_config.add_child(self._cup_y_slider)
        self.cup_y = 0 
        
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

        self._settings_panel.add_child(inference_config)
        
        # add scene
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # # initial scene
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1)
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        
        # initial scene
        self.traj = self.ds.traj_data[0]
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
    
    def add_env(self):
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)
        # self._scene.scene.add_geometry('obs_init', self.mesh_obs, self.mat)
        
        mesh_mug_ = deepcopy(self.mesh_mug)
        mesh_mug_.translate([self.cup_x, self.cup_y, 0])
        self._scene.scene.add_geometry('mug_init', mesh_mug_, self.mat)
        
        # update trajectory
        # mesh_bottle_ = deepcopy(self.mesh_bottle)
        # T_init = self.ds.traj_data[0][0]
        # mesh_bottle_.transform(T_init)
        # if self.draw_bottle_label:
            # mesh_bottle_label_ = deepcopy(self.mesh_bottle_label)
            # mesh_bottle_label_.transform(T_init)

        # self._scene.scene.add_geometry(f'bottle_init', mesh_bottle_, self.mat)
        # if self.draw_bottle_label:
            # self._scene.scene.add_geometry(f'bottle_label_init', mesh_bottle_label_, self.mat)
        
    def _set_vis_mode_video(self):
        self.reset_threads()
        self.thread_traj.start()

    def _set_vis_mode_afterimage(self):
        self.reset_threads()
        self._scene.scene.clear_geometry()
        self.add_env()
        self.update_trajectory()
        
    def _set_sample_idx(self, value):
        self.traj_idx = int(value)
        self.traj = self.trajs[self.traj_idx]
        self._set_vis_mode_afterimage()
    
    def update_trajctory(self):
        traj_data = self.ds.traj_data
        w = self.mmp.get_w_from_traj(traj_data.to(self.device))
        zs = self.mmp.encode(w)

        idx = int(4*(self.latent_value + 1)/2)
        if idx == 4:
            selected_z = zs[idx]
        else:
            t = 4*(self.latent_value + 1)/2 - idx
            selected_z = (1-t)*zs[idx:idx+1] + t*zs[idx+1:idx+2]
            
        selected_w = self.mmp.decode(selected_z)
        selected_w = selected_w.view(-1, self.mmp.b+1, 6)
        selected_w[:, 0, 3] = selected_w[:, 0, 3] + self.cup_x
        selected_w[:, 0, 4] = selected_w[:, 0, 4] + self.cup_y
        selected_w = selected_w.view(-1, 6*(self.mmp.b+1))
        self.traj = self.mmp.w_to_SE3(selected_w, traj_len=480)[0].detach().cpu()
    
    def _set_cup_x(self, value):
        self.cup_x = float(value)
        self.update_trajctory()
        self._set_vis_mode_afterimage()
    
    def _set_cup_y(self, value):
        self.cup_y = float(value)
        self.update_trajctory()
        self._set_vis_mode_afterimage()
        
    def _set_latent_coordinates(self, value):
        self.latent_value = float(value)
        self.update_trajctory()
        self._set_vis_mode_afterimage()
    
    def reset_threads(self):
        if self.thread_traj.is_alive():
            self.end_thread(self.thread_traj)
        self.thread_traj = threading.Thread(
                            target=self.update_trajectory_video,
                            daemon=True)
        
    def update_trajectory(self):
        for idx in range(0, len(self.traj), 5*self.skip_size):
            mesh_bottle_ = deepcopy(self.mesh_bottle)
            T = self.traj[idx]
            mesh_bottle_.transform(T)
            if self.draw_bottle_label:
                mesh_bottle_label_ = deepcopy(self.mesh_bottle_label)
                mesh_bottle_label_.transform(T)
            self._scene.scene.add_geometry(f'bottle_{idx}', mesh_bottle_, self.mat)
            if self.draw_bottle_label:
                self._scene.scene.add_geometry(f'bottle_label_{idx}', mesh_bottle_label_, self.mat)

        # # include last
        # idx = len(self.traj) - 1
        # mesh_bottle_ = deepcopy(self.mesh_bottle)
        # T = self.traj[idx]
        # mesh_bottle_.transform(T)
        # if self.draw_bottle_label:
        #     mesh_bottle_label_ = deepcopy(self.mesh_bottle_label)
        #     mesh_bottle_label_.transform(T)
        # self._scene.scene.add_geometry(f'bottle_{idx}', mesh_bottle_, self.mat)
        # if self.draw_bottle_label:
        #     self._scene.scene.add_geometry(f'bottle_label_{idx}', mesh_bottle_label_, self.mat)
                
    def update_trajectory_video(self):
        # update trajectory
        self.flag_update_scene = [False] * len(self.traj)
        
        self.mesh_bottle_ = deepcopy(self.mesh_bottle)
        if self.draw_bottle_label:
            self.mesh_bottle_label_ = deepcopy(self.mesh_bottle_label)
        
        self.mesh_bottle_.transform(self.traj[0])
        self.mesh_bottle_label_.transform(self.traj[0])
        
        for idx in range(1, len(self.traj)):
            tic = time.time()
            if self.event.is_set():
                break
            self.idx = idx
            
            # Update geometry
            gui.Application.instance.post_to_main_thread(self.window, self.update_scene)
            # wait for dt
            dt = 0.002 
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
        T = self.traj[self.idx]
        T_prev = self.traj[self.idx-1]
        self.mesh_bottle_.transform(T@inverse_SE3(T_prev.unsqueeze(0)).squeeze(0))
        self.mesh_bottle_label_.transform(T@inverse_SE3(T_prev.unsqueeze(0)).squeeze(0))
        self._scene.scene.add_geometry(f'bottle_{self.idx}', self.mesh_bottle_, self.mat)
        if self.draw_bottle_label:
            self._scene.scene.add_geometry(f'bottle_label_{self.idx}', self.mesh_bottle_label_, self.mat)
        self.flag_update_scene[self.idx] = True
        
if __name__ == "__main__":

    gui.Application.instance.initialize()

    w = AppWindow()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()