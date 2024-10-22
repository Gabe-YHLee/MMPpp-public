import os
import threading
import pybullet as p
import pybullet_data
import math
import numpy as np
import time
import copy
from scipy.spatial.transform import Rotation as R
from utils.lie import *

from utils.util_shelf import (
    shelf_sampler, 
    box_urdf_writer,
    shelf_urdf_writer, 
    shelf_surface_points_sampler, 
    suface_points_sampler_box,
    shelf_candidate_3dgrid_points,
    sample_two_points,
    sample_random_gripper_orientations
)

class Point2PointTaskEnv:
    def __init__(
            self, 
            robot_path):
        
        # environment settings
        self.plane_z = -0.8
        self._robot_home_joint_config = [
            0.0301173169862714, -1.4702106391932968, 0.027855688427362513, 
            -2.437557753144649, 0.14663284881434122, 2.308719465520647, 
            0.7012385825324389]

        # add path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        # add ground plane
        self._plane_id = p.loadURDF("plane.urdf", [0, 0, self.plane_z])
        
        # add robot arm
        self._robot_body_id = p.loadURDF(robot_path, [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(p.getNumJoints(self._robot_body_id))]
        self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._robot_joint_lower_limit = [x[8] + 0.0037 for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._robot_joint_upper_limit = [x[9] - 0.0037 for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._finger_joint_indices = [8, 9]
        joint_limit_offset = 0.02
        self._joint_epsilon = joint_limit_offset/2
        self._robot_EE_joint_idx = 7
        self._robot_tool_joint_idx = 9
        self._robot_tool_tip_joint_idx = 9
        
        # robot last link to ee
        ee_pose6 = self.state2pose(p.getLinkState(self._robot_body_id, 6))
        ee_pose7 = self.state2pose(p.getLinkState(self._robot_body_id, 7))
        self.LastLink2EE = inv_SE3(ee_pose6)@ee_pose7
         
        # Set friction coefficients for gripper fingers
        p.changeDynamics(
            self._robot_body_id, 7,
            lateralFriction=1, # 0.1
            spinningFriction=1, # 0.1
            rollingFriction=1,
            frictionAnchor=True
        )
        p.changeDynamics(
            self._robot_body_id, 8,
            lateralFriction=0.1, # 0.1
            spinningFriction=0.1, # 0.1
            rollingFriction=0.1,
            frictionAnchor=True
        )
        p.changeDynamics(
            self._robot_body_id, 9,
            lateralFriction=0.1, # 0.1
            spinningFriction=0.1, # 0.1
            rollingFriction=0.1,
            frictionAnchor=True
        )
        
        # # camera
        # self.kinect_pose = np.array([[0., 0., 1., - 0.75],
        #                              [-1., 0., 0., 0.],
        #                              [0., -1. ,0., 0.71016075],
        #                              [0, 0, 0, 1]])
        
        # self.kinect_intrinsic = np.array(
        #     [609.7949829101562, 609.4755859375, 640.93017578125, 368.19635009765625]
        # ) # fx, fy, px, py

        # # camera list
        # self.camera_params = {
        #     # azure kinect
        #     0: self._get_camera_param(
        #         camera_pose = self.kinect_pose,
        #         camera_intrinsic = self.kinect_intrinsic,
        #         camera_image_size=[720, 1280]
        #     ),
        # }
        # self.camera_freq = 40

        # reset debug camera view
        p.resetDebugVisualizerCamera(2.000002861022949, -135.60009765625, -33.79999923706055, (0, 0, 0))

        # go home position
        self.init_simulation()
        
    def init_simulation(self):
        self.reset_robot()
        self.reset_gripper()
        self.timeState = 0.0

    def state2pose(self, state):
        ee_rot = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3,3)
        ee_position = np.array(state[4])
        return np.vstack([
            np.hstack([ee_rot, ee_position.reshape(3, 1)]),
            np.array([0, 0, 0, 1])])
         
    def add_shelf(
        self,
        shelf_path,
        shelf_position,
        shelf_orientation,
    ):
        shelf_id = p.loadURDF(
            shelf_path, 
            shelf_position, 
            shelf_orientation, 
            useFixedBase=True)
        return shelf_id
    
    def add_target_ee(
        self, 
        ee_path,
        basePosition=[0, 0, 2],
        baseOrientation=[0, 0, 0, 1],
        rgbaColor=[1.0, 0.5, 0.5, 0.5]
        ):
        ee_id = p.loadURDF(ee_path, basePosition, baseOrientation, useFixedBase=True)
        if rgbaColor is not None:
            for i in [0, 1, -1]:
                p.changeVisualShape(ee_id, i, rgbaColor=rgbaColor)

        target_width = 0.04
        for i in [0, 1]:
            p.resetJointState(ee_id, i, target_width)
        return ee_id
    
    def run(self):
        # add thread
        step_sim_thread = threading.Thread(target=self.simulation)
        step_sim_thread.daemon = True
        step_sim_thread.start()
        
    def simulation(self, ts=1./240.):
        p.setTimeStep(ts)
        while True:
            p.stepSimulation()
            time.sleep(ts)
            self.timeState += ts

    def add_box2ee(self, size_x, size_y, size_z):
        ee_state = p.getLinkState(self._robot_body_id, 7)
        ee_rot = np.asarray(p.getMatrixFromQuaternion(ee_state[5])).reshape(3,3)
        position = np.array(ee_state[4]) + (ee_rot@np.array([0, 0, 0.1]).reshape(3, 1)).reshape(-1)
        orientation = (R.from_matrix(ee_rot@np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))).as_quat()
           
        box_id = self._create_box(size_x, size_y, size_z, position, orientation)
        return box_id
    
    def add_cylinder2ee(self, size_r, size_h):
        ee_state = p.getLinkState(self._robot_body_id, 7)
        ee_rot = np.asarray(p.getMatrixFromQuaternion(ee_state[5])).reshape(3,3)
        position = np.array(ee_state[4]) + (ee_rot@np.array([0, 0, 0.1]).reshape(3, 1)).reshape(-1)
        orientation = (R.from_matrix(ee_rot@np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))).as_quat()
               
        cylinder_id = self._create_cylinder(
            size_r, 
            size_h, 
            position, 
            orientation)
        
        self.grasp_object(
            blocking=True,
            force=1)
        return cylinder_id
    
    def grasp_object(self, blocking=False, force=200):
        
        # target joint state
        target_joint_state = np.array([0.0, 0.0])
        forces = np.array([force, force])
        
        # Move joints
        p.setJointMotorControlArray(
            self._robot_body_id, 
            self._finger_joint_indices,
            p.POSITION_CONTROL, 
            target_joint_state,
            forces=forces,
            # positionGains=speed * np.ones(len(self._finger_joint_indices))
        )

        # Block call until joints move to target configuration
        if blocking:
            timeout_t0 = time.time()
            while True:
                if time.time() - timeout_t0 > 1:
                    break
                time.sleep(0.0001)
    
    def _create_cylinder(self, size_r, size_h, position, orientation):      
        # declare collision
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=size_r, 
            height=size_h
        )
        
        # create object
        body_id = p.createMultiBody(
            0.05, 
            collision_id, 
            -1, 
            position, 
            p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation
        )
        p.changeDynamics(
            body_id, 
            -1, 
            spinningFriction=0.002, 
            lateralFriction=0.4, 
            mass=0.585*np.pi*size_r*size_r*size_h/(125*1000)
        )
        p.changeVisualShape(
            body_id, 
            -1, 
            rgbaColor=np.concatenate([1 * np.random.rand(3), [1]])
        )
        # self.object_ids.append(body_id)
        # self.voxel_coord[body_id] = coord / scale_cylinder
        
        # # open3d mesh
        # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=size_r/scale_cylinder, height=size_h/scale_cylinder, resolution=100, split=10)
        # self.meshes[body_id] = mesh_cylinder
        # object_info = {'type': 'cylinder', 'size': [size_r/scale_cylinder, size_r/scale_cylinder, size_h/scale_cylinder]}
        # self.object_info.append(object_info)
        return body_id

    def _create_box(self, size_x, size_y, size_z, position, orientation, enable_stacking=False):             
        # # voxelize object
        # md = np.ones([size_x, size_y, size_z])
        # coord = (np.asarray(np.nonzero(md)).T + 0.5 - np.array([size_x/2, size_y/2, size_z/2]))

        # declare collision
        collision_id = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=np.array(
                [size_x/2, size_y/2, size_z/2]
            )
        )
        
        # create object
        body_id = p.createMultiBody(
            0.05, 
            collision_id, 
            -1, 
            position, 
            p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation
        )
        p.changeDynamics(
            body_id, 
            -1, 
            spinningFriction=0.002 if not enable_stacking else 0.2, 
            lateralFriction=0.4 if not enable_stacking else 0.6, 
            mass=1 #0.737*size_x*size_y*size_z/(125*1000)
        )
        p.changeVisualShape(
            body_id, 
            -1, 
            rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))
        
        # self.object_ids.append(body_id)
        # self.voxel_coord[body_id] = coord

        # # open3d mesh
        # mesh_box = o3d.geometry.TriangleMesh.create_box(
        #     width = size_x,
        #     height = size_y, 
        #     depth = size_z
        # )
        # mesh_box.translate([-size_x/(2), -size_y/(2), -size_z/(2)]) # match center to the origin
        # self.meshes[body_id] = mesh_box
        # object_info = {
        #     'type': 'box', 
        #     'size': [size_x, size_y, size_z]
        # }
        # self.object_info.append(object_info)
        return body_id

    def _get_camera_param(
            self, 
            camera_pose,
            camera_intrinsic,
            camera_image_size
        ):

        # modified camera intrinsic
        fx = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
        fy = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
        px = float(camera_image_size[1]) / 2
        py = float(camera_image_size[0]) / 2

        # camera view matrix
        camera_view_matrix = copy.deepcopy(camera_pose)
        camera_view_matrix[:, 1:3] = -camera_view_matrix[:, 1:3]
        camera_view_matrix = np.linalg.inv(camera_view_matrix).T.reshape(-1)
        
        # camera z near/far values (arbitrary value)
        camera_z_near = 0.01
        camera_z_far = 20

        # # camera intrinsic matrix
        # camera_intrinsic_matrix = np.array(
        # 	[[camera_intrinsic[0], 0, camera_intrinsic[2]],
        # 	 [0, camera_intrinsic[1], camera_intrinsic[3]],
        # 	 [0, 0, 1]]
        # )

        # camera intrinsic matrix
        camera_intrinsic_matrix = np.array(
            [[fx, 0, px],
             [0, fy, py],
             [0, 0, 1]]
        )

        # camera projection matrix
        camera_fov_h = (math.atan(py / fy) * 2 / np.pi) * 180
        camera_projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera_fov_h,
            aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
            nearVal=camera_z_near,
            farVal=camera_z_far
        )  

        camera_param = {
            'camera_image_size': camera_image_size,
            'camera_intr': camera_intrinsic_matrix,
            'camera_pose': camera_pose,
            'camera_view_matrix': camera_view_matrix,
            'camera_projection_matrix': camera_projection_matrix,
            'camera_z_near': camera_z_near,
            'camera_z_far': camera_z_far
        }
        return camera_param

    def get_camera_data(self, cam_param):
        camera_data = p.getCameraImage(cam_param['camera_image_size'][1], cam_param['camera_image_size'][0],
                                       cam_param['camera_view_matrix'], cam_param['camera_projection_matrix'],
                                       shadow=1, renderer=p.ER_TINY_RENDERER)
        color_image = np.asarray(camera_data[2]).reshape(
            [cam_param['camera_image_size'][0], cam_param['camera_image_size'][1], 4]
        )[:, :, :3]  # remove alpha channel
        z_buffer = np.asarray(camera_data[3]).reshape(cam_param['camera_image_size'])
        camera_z_near = cam_param['camera_z_near']
        camera_z_far = cam_param['camera_z_far']
        depth_image = (2.0 * camera_z_near * camera_z_far) / (
            camera_z_far + camera_z_near - (2.0 * z_buffer - 1.0) * (
                camera_z_far - camera_z_near
            )
        )
        mask_image = np.asarray(camera_data[4]).reshape(cam_param['camera_image_size'][0:2])
        return color_image, depth_image, mask_image

    def robot_go_home(self, blocking=True, speed=0.1):
        self.move_joints(self._robot_home_joint_config, blocking, speed)

    def reset_robot(self, jointPos=None):
        if jointPos is None:
            jointPos = self._robot_home_joint_config
        for i in self._robot_joint_indices:
            p.resetJointState(self._robot_body_id, i, jointPos[i])
    
    def reset_gripper(self):
        for i in self._finger_joint_indices:
            p.resetJointState(self._robot_body_id, i, 0.04)
                
    def move_joints(self, target_joint_state, blocking=False, positionGains=[0.01]*7, velocityGains=[0.2]*7):
        # move joints
        p.setJointMotorControlArray(
            self._robot_body_id, 
            self._robot_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_joint_state,
            targetVelocities=[0]*7,
            positionGains=positionGains, 
            velocityGains=velocityGains
           )

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):
                if time.time() - timeout_t0 > 3:
                    return False
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
                time.sleep(0.0001)
            return True
        else:
            return 1

    def move_gripper(self, target_width, blocking=False, speed=0.03):
        # target joint state
        target_joint_state = np.array([target_width, target_width])
        
        # Move joints
        p.setJointMotorControlArray(
            self._robot_body_id, 
            self._finger_joint_indices,
            p.POSITION_CONTROL, 
            target_joint_state,
            positionGains=speed * np.ones(len(self._finger_joint_indices))
        )

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in range(len(actual_joint_state))]):
                if time.time() - timeout_t0 > 3:
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
                time.sleep(0.0001)


class RobotEnv(Point2PointTaskEnv):
    def __init__(
        self, 
        robot_path):
        super(RobotEnv, self).__init__(robot_path=robot_path)
        
        # gen shelf
        shelf_path1 = "assets/shelf/shelf1.urdf" 
        shelf_urdf_writer(
            shelf_path1, 0.4, 0.7, 0.7+0.4, np.array([0.1]), np.array([0.4+0.4]), 0.02)
        shelf_surface_points1 = shelf_surface_points_sampler(
            4096, 0.4, 0.7, 0.7+0.4, np.array([0.1]), np.array([0.4+0.4]), 0.02)
        shelf_path2 = "assets/shelf/shelf2.urdf" 
        shelf_urdf_writer(
            shelf_path2, 0.4, 0.5, 0.8+0.4, np.array([]), np.array([0.25+0.3]), 0.02)
        shelf_surface_points2 = shelf_surface_points_sampler(
            4096, 0.4, 0.5, 0.8+0.4, np.array([]), np.array([0.25+0.3]), 0.02)
        shelf_table_path = "assets/shelf/table.urdf"
        box_urdf_writer(
            shelf_table_path, 0.4, 0.4, 0.02
        )
        table_surface_points3 = suface_points_sampler_box(512, 0, 0, 0, 0.4, 0.4, 0.02)
        shelf_table_path2 = "assets/shelf/table2.urdf"
        box_urdf_writer(
            shelf_table_path2, 0.4, 0.4, 0.6
        )
        table_surface_points4 = suface_points_sampler_box(512, 0, 0, 0, 0.4, 0.4, 0.6)
        
        # add shelf
        shelf1_position = [0.7, 0.0, 0.0]
        shelf1_orientation = [0.0, 0.0, 1., 0.]
        shelf1_rot = np.asarray(
            p.getMatrixFromQuaternion(shelf1_orientation)).reshape(3, 3)
        
        shelf_surface_points1 = (
            shelf1_rot@shelf_surface_points1.T + np.array(shelf1_position).reshape(3, 1)).T
        
        shelf1_id = self.add_shelf(
            shelf_path1,
            shelf_position=shelf1_position,
            shelf_orientation=shelf1_orientation
        )
        
        shelf2_position = [0.0, -0.7, 0.0]
        shelf2_orientation = [0.0, 0.0, 1., 0.]
        shelf2_rot = np.asarray(
            p.getMatrixFromQuaternion(shelf2_orientation)).reshape(3, 3)@np.array([
                    [0, 1, 0], 
                    [-1, 0, 0], 
                    [0, 0, 1]])
        shelf2_orientation = (
            R.from_matrix(
                shelf2_rot
                )
            ).as_quat()
        
        shelf_surface_points2 = (
            shelf2_rot@shelf_surface_points2.T + np.array(shelf2_position).reshape(3, 1)).T
        
        shelf2_id = self.add_shelf(
            shelf_path2,
            shelf_position=shelf2_position,
            shelf_orientation=shelf2_orientation
        )
        
        shelf_table_position = [0.45, -0.3, 0.39+0.4]
        shelf_table_orientation = [0.0, 0.0, 1., 0.]
        shelf_table_rot = np.asarray(
            p.getMatrixFromQuaternion(shelf_table_orientation)).reshape(3, 3)
        
        table_surface_points3 = table_surface_points3 + np.array([shelf_table_position])
        
        shelf_table_orientation = (
            R.from_matrix(
                shelf_table_rot@np.array([
                    [1, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 1]])
                )
            ).as_quat()
            
        shelf_table_id = self.add_shelf(
            shelf_table_path,
            shelf_position=shelf_table_position,
            shelf_orientation=shelf_table_orientation
        )
        
        shelf_table2_position = [0.45, -0.7, 0.5]
        table_surface_points4 = table_surface_points4 + np.array([shelf_table2_position])
        shelf_table2_orientation = [0.0, 0.0, 1., 0.]
        shelf_table2_rot = np.asarray(
            p.getMatrixFromQuaternion(shelf_table2_orientation)).reshape(3, 3)
        shelf_table2_orientation = (
            R.from_matrix(
                shelf_table2_rot@np.array([
                    [1, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 1]])
                )
            ).as_quat()
            
        shelf_table2_id = self.add_shelf(
            shelf_table_path2,
            shelf_position=shelf_table2_position,
            shelf_orientation=shelf_table2_orientation
        )
        
        self.all_env_surface_points = np.concatenate(
            [
                shelf_surface_points1,
                shelf_surface_points2,
                table_surface_points3,
                table_surface_points4
            ], axis=0
        )

            
        ############################################
        ############################################
        ############################################
        ########### EE POSE SAMPLE #################
        ############################################
        ############################################
        ############################################
        
        init_ee_position = np.array([0.5, 0.15, 0.6+0.3])
        init_ee_rot = np.array([
                    [-1, 0, 0], 
                    [0, -1, 0], 
                    [0, 0, 1]])@np.array([
                    [0, 0, -1], 
                    [0, 1, 0], 
                    [1, 0, 0]])
        init_ee_id = self.add_target_ee(
            'assets/panda/gripper_vis_only.urdf',
            basePosition=init_ee_position,
            baseOrientation=R.from_matrix(init_ee_rot).as_quat(),
            rgbaColor = [1.0, 0.5, 0.5, 0.5]
            )
        
        init_ee_pose = np.vstack([
                np.hstack([init_ee_rot, init_ee_position.reshape(3, 1)]), 
                np.array([0, 0, 0, 1])])
        # init_converge_flag, q_i = panda.solveInverseKinematics(
        #     init_ee_pose@inv_SE3(self.LastLink2EE), 
        #     init_jointPos=self._robot_home_joint_config,
        #     jointLowerLimits=np.array(self._robot_joint_lower_limit),
        #     jointUpperLimits=np.array(self._robot_joint_upper_limit)
        #     )
        # print(init_converge_flag, q_i) 
        
        q_i =  [-1.22937435, -0.3774999, 1.17831243, -1.4427563, 0.28473847, 2.87936954, 0.85840966]
        for i in self._robot_joint_indices:
            p.resetJointState(
                self._robot_body_id, 
                i, 
                q_i[i])
        
        final_ee_position = np.array([0.0, -0.5, 0.65])
        final_ee_rot = np.array([
                    [0, 1, 0], 
                    [-1, 0, 0], 
                    [0, 0, 1]])@np.array([
                    [-1, 0, 0], 
                    [0, -1, 0], 
                    [0, 0, 1]])@np.array([
                    [0, 0, -1], 
                    [0, 1, 0], 
                    [1, 0, 0]])
        final_ee_id = self.add_target_ee(
            'assets/panda/gripper_vis_only.urdf',
            basePosition=final_ee_position,
            baseOrientation=R.from_matrix(final_ee_rot).as_quat(),
            rgbaColor = [0.5, 0.5, 1, 0.5]
            )
        
        final_ee_pose = np.vstack([
                np.hstack([final_ee_rot, final_ee_position.reshape(3, 1)]), 
                np.array([0, 0, 0, 1])])
        # final_converge_flag, q_f = panda.solveInverseKinematics(
        #     final_ee_pose@inv_SE3(self.LastLink2EE), 
        #     init_jointPos=self._robot_home_joint_config,
        #     jointLowerLimits=np.array(self._robot_joint_lower_limit),
        #     jointUpperLimits=np.array(self._robot_joint_upper_limit)
        #     )
        # print(final_converge_flag, q_f) 
        # q_f = [-1.58767295,  0.38751211,  0.56713907, -0.87841404,  1.98681289,  3.69838667, -1.49820622]
        # for i in self._robot_joint_indices:
        #     p.resetJointState(
        #         self._robot_body_id, 
        #         i, 
        #         q_f[i])
        
        q_f = [-0.96280892, -0.2466983, -0.46373697, -2.23105282, 0.34272699, 3.62121556, 0.36830725]