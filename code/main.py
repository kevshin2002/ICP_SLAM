from data.dataset import Dataset
from utils.tester import Tester
from utils.timer import Timer
from utils.plotter import Plotter
from core.driver import Driver
import numpy as np
import open3d as o3d
import os
import logging

if __name__ == "__main__":
    ###########################################################################
    #                                  Data                                   #
    #                      Encoder, Lidar, IMU, Pointcloud                    #
    ###########################################################################
    DATA_PATH = "../data/" # change these later to be cli or yaml vars
    SET = 21 
    NUM = 4
    MODEL = "drill"
    dataset = Dataset(PATH=DATA_PATH, DATASET=SET, MODEL_NAME=MODEL, PC_NUM=NUM)
    encoder = dataset.get_encoder_data()

    logging.info("Fetching Encoder Data")
    logging.info(f"Encoder Counts Shape: {encoder['counts'].shape}")
    logging.info(f"Encoder Stamps Shape: {encoder['stamps'].shape}")
    
    lidar = dataset.get_lidar_data()
    logging.info("Fetching Lidar Data")
    logging.info(f"Lidar Range Shape: {lidar['ranges'].shape}")
    logging.info(f"Lidar Angle Min: {lidar['angle_min']}")
    logging.info(f"Lidar Angle Max: {lidar['angle_max']}")
    logging.info(f"Lidar Time Stamps: {lidar['stamps'].shape}")

    imu = dataset.get_imu_data()
    logging.info("Fetching IMU Data")
    logging.info(f"IMU Angular Velocity Shape: {imu['angular_velocity'].shape}")
    logging.info(f"IMU Linear Acceleration Shape: {imu['linear_acceleration'].shape}")
    logging.info(f"IMU Stamps Shape: {imu['stamps'].shape}")
    
    pc = dataset.get_pointcloud_data()
    logging.info("Fetching PointCloud Data")
    #logging.info(f"PointCloud Source and Target Shape: {pc['source'].shape}, {pc['target'].shape}")
    
    ###########################################################################
    #                            Implementation Steps                         #
    #                 Motion Model via Differential Drive Kinematic           #
    #                       Improvement via LIDAR and ICP                     #
    #                         2D Occupancy Grid Map                           #
    #                  GTSAM Optimization (Loop Closures / Factors)           #
    ###########################################################################
    TRAJECTORY_PATH = os.path.join('../output/trajectory/', f"{SET}.png")
    driver = Driver(encoder, imu, lidar)
    model = driver.modelize()
    Plotter.view_trajectory(model, output_path=TRAJECTORY_PATH)
    SOURCE = pc['source']
    TRANSFORM = np.eye(4)

    for TARGET_NUM in range(NUM):
        TARGET = pc['target'][TARGET_NUM]
        DIRECTORY_PATH = os.path.join('../output/', MODEL, str(TARGET_NUM))
        T, errors = driver.icp(SOURCE, TARGET, TRANSFORM, output_path=DIRECTORY_PATH)
    optimized_model = driver.scan(model)
    OPTIMIZED_PATH = os.path.join('../output/trajectory/', f"optimized_{SET}.png")
    COMPARISON_PATH = os.path.join('../output/trajectory/', f"comparison_{SET}.png")
    Plotter.view_trajectory(optimized_model, output_path=OPTIMIZED_PATH)
    Plotter.compare_trajectory(model, optimized_model, output_path=COMPARISON_PATH)

