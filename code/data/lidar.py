import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Lidar:
    def __init__(self, PATH: str, DATASET: int):
        self.DATASET = DATASET
        logging.info(f"Initializing Lidar with dataset {DATASET} from path {PATH}Hokuyo{DATASET}")
        logging.info(f"Attempting to load Lidar data for dataset {DATASET}.")
        with np.load(f"{PATH}Hokuyo{DATASET}.npz") as data:
            self.lidar_angle_min = data["angle_min"]
            self.lidar_angle_max = data["angle_max"]
            self.lidar_angle_increment = data["angle_increment"]
            self.lidar_range_min = data["range_min"]
            self.lidar_range_max = data["range_max"]
            self.lidar_ranges = data["ranges"]
            self.lidar_stamps = data["time_stamps"]
            logging.info(f"Successfully loaded Lidar data for dataset {DATASET}.")

    def get_data(self):
        logging.info(f"Accessing Lidar data for dataset {self.DATASET}.")
        return {
            "angle_min": self.lidar_angle_min,
            "angle_max": self.lidar_angle_max,
            "angle_increment": self.lidar_angle_increment,
            "range_min": self.lidar_range_min,
            "range_max": self.lidar_range_max,
            "ranges": self.lidar_ranges,
            "stamps": self.lidar_stamps,
        }

