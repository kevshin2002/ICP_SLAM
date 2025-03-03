import numpy as np
import logging
import matplotlib.pyplot as plt
import os
import open3d as o3d
import gtsam
from tqdm import tqdm
from scipy.linalg import expm
from sklearn.neighbors import NearestNeighbors
from utils.plotter import Plotter
from utils.helper import Helper

class Driver:
    def __init__(self, encoder: dict, imu: dict, lidar: dict):
        RIGHT_COUNT = (encoder['counts'][1, 1:] + encoder['counts'][3, 1:]) / 2
        LEFT_COUNT = (encoder['counts'][0, 1:] + encoder['counts'][2, 1:]) / 2
        self.DT = np.diff(encoder['stamps'])
        RADIUS = 0.127
        RESOLUTION = 360
        CIRCUMFERENCE = 2 * np.pi * RADIUS
        CONVERSION = CIRCUMFERENCE / RESOLUTION

        VR = (RIGHT_COUNT * CONVERSION) / self.DT
        VL = (LEFT_COUNT * CONVERSION) / self.DT
        self.V = (VR + VL) / 2

        imu_ipt = np.cumsum(np.concatenate(([0], self.DT)))
        yaw_ipt = np.linspace(imu_ipt[0], imu_ipt[-1], imu['angular_velocity'].shape[1])
        self.YAW = np.interp(imu_ipt[:-1], yaw_ipt, imu['angular_velocity'][2, :])
        
        indices = np.searchsorted(encoder['stamps'], lidar['stamps']) - 1 
        indices = np.clip(indices, 0, len(encoder['stamps']) - 1)
        ANGLES = lidar['angle_min'] + np.arange(lidar['ranges'].shape[0]) * lidar['angle_increment']

        x = lidar['ranges'] * np.cos(ANGLES).reshape(-1, 1)
        y = lidar['ranges'] * np.sin(ANGLES).reshape(-1, 1)
        self.lidar = np.ones((x.shape[0], x.shape[1], 3))
        self.lidar[:, :, 0] = x
        self.lidar[:, :, 1] = y
        to_body = np.array([[1, 0, 0.30183], 
                            [0, 1, 0],
                            [0, 0, 1]
                            ])
        self.body = self.lidar
        self.body = np.zeros_like(self.lidar)
        for i in range(self.lidar.shape[1]):
            self.body[:, i, :] = (to_body @ self.lidar[:, i, :].T).T  

    def modelize(self):
        poses = np.zeros((len(self.DT) + 1, 4, 4))
    
        pose = np.eye(4)    
        for i in range(1, len(self.V) + 1):
            dt_i = self.DT[i-1]
            v = self.V[i-1]
            omega = self.YAW[i-1]
        
            twist = np.zeros((4, 4))
            twist[0, 1] = -omega
            twist[1, 0] = omega
            twist[0, 3] = v
            twist[1, 3] = 0
            
            pose = pose @ expm(twist * dt_i)
            poses[i] = pose

            poses[i, 0, 3] = pose[0, 3]  # x position
            poses[i, 1, 3] = pose[1, 3]  # y position
            poses[i, 2, 3] = np.arctan2(pose[1, 0], pose[0, 0])  # theta        
            poses[i, 2, 3] = np.arctan2(np.sin(poses[i, 2, 3]), np.cos(poses[i, 2, 3]))

        return poses

    def kabsch(self, source, target):
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)

        source_centered = (source - source_centroid)
        target_centered = (target - target_centroid)
        
        Q = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(Q)
        scale = np.eye(3)
        det = np.linalg.det(U @ Vt)
        scale[-1, -1] = det
        R = U @ scale @ Vt
        p = source_centroid - (R @ target_centroid)
        return R, p

    def icp(self, source, target, transform, epochs=100, tolerance=1e-8, output_path=None):
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        theSource = source.copy()
        theTarget = target.copy()
        T = transform
        source = source @ T[:3, :3]
        prev_error = float('inf')
        errors = []
        
        progress_bar = tqdm(range(epochs), desc='Optimizing via Iteractive Closest Point')
        
        for iteration in progress_bar:
            distances, indices = tuple(arr.ravel() for arr in NearestNeighbors(n_neighbors=1).fit(theTarget).kneighbors(source))
            error = np.mean(np.linalg.norm(theTarget[indices] - source, axis=1))
            errors.append(error)
            progress_bar.set_description(f'ICP Progress (error: {error:.6f})')
            if abs(prev_error - error) < tolerance:
                break
            R, p = self.kabsch(source, theTarget[indices])
            theTarget = (R @ theTarget.T).T + p 

            current_T = np.eye(4)
            current_T[:3, :3] = R
            current_T[:3, 3] = p
            T = current_T @ T
        
            if output_path is not None and iteration % 10 == 0:
                Plotter.view_icp(source, target, T, os.path.join(output_path, f'progress_{iteration:1d}.jpg')) 
            prev_error = error 
        progress_bar.close()
        if output_path is not None:
            Plotter.multiview(source, target, T, output_path)
        return T, errors

    def scan(self, poses):
        do = np.diff(poses, axis=0)
        dx = do[:, 0, 3]
        dy = do[:, 1, 3]
        dt = do[:, 2, 3]
        
        optimized = []
        T = np.eye(4)
        initial  = np.zeros((poses.shape[0], 4, 4))
        for theIteration in range(do.shape[0]):
            R = np.array([
                        [np.cos(dt[theIteration]), -np.sin(dt[theIteration]), 0],
                        [np.sin(dt[theIteration]), np.cos(dt[theIteration]), 0],
                        [0, 0, 1]
                        ])
            p = np.array([dx[theIteration], dy[theIteration], 0])
        
            initial[theIteration, :3, :3] = R
            initial[theIteration, :3, 3] = p
            initial[theIteration, 3, 3] = 1

        print(self.body.shape[1])
        print(poses.shape)
        for theIteration in range(poses.shape[0]):
            if theIteration + 1 >= poses.shape[0]:
                break 

            relative_pose, _ = self.icp(self.body[:, theIteration, :], self.body[:, theIteration+1, :], initial[theIteration])
            T = T @ relative_pose
            optimized.append(T)

        return np.array(optimized)

