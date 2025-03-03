from .encoder import Encoder
from .imu import IMU
from .lidar import Lidar
from .pointcloud import PointCloud
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dataset:
    def __init__(self, PATH: str, DATASET: int, MODEL_NAME: str, PC_NUM: int):
        self.encoder = Encoder(PATH, DATASET)
        self.lidar = Lidar(PATH, DATASET)
        self.imu = IMU(PATH, DATASET)
        self.pointcloud = PointCloud(PATH, MODEL_NAME, PC_NUM)

    def get_encoder_data(self):
        return self.encoder.get_data()

    def get_lidar_data(self):
        return self.lidar.get_data()

    def get_imu_data(self):
        return self.imu.get_data()

    def get_pointcloud_data(self):
        return self.pointcloud.get_data()

