import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IMU:
    def __init__(self, PATH: str, DATASET: int):
        self.DATASET = DATASET
        logging.info(f"Initializing IMU with dataset {DATASET} from path {PATH}Imu{DATASET}")
        logging.info(f"Attempting to load IMU data for dataset {DATASET}.")
        with np.load(f"{PATH}Imu{DATASET}.npz") as data:
            self.imu_angular_velocity = data["angular_velocity"]
            self.imu_linear_acceleration = data["linear_acceleration"]
            self.imu_stamps = data["time_stamps"]
            logging.info(f"Successfully loaded IMU data for dataset {DATASET}.")

    def get_data(self):
        logging.info(f"Accessing IMU data for dataset {self.DATASET}.")
        return {
            "angular_velocity": self.imu_angular_velocity,
            "linear_acceleration": self.imu_linear_acceleration,
            "stamps": self.imu_stamps,
        }

