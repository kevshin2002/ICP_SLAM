import numpy as np
import logging
from utils.logging_config import logger

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Encoder:
    def __init__(self, PATH: str, DATASET: int):
        self.DATASET = DATASET
        logging.info(f"Initializing Encoder with dataset {DATASET} from path {PATH}Encoder{DATASET}.npz")
        logging.info(f"Attempting to load encoder data for dataset {DATASET}.")
        with np.load(f"{PATH}Encoders{DATASET}.npz") as data:
            self.encoder_counts = data["counts"]
            self.encoder_stamps = data["time_stamps"]
            logging.info(f"Successfully loaded encoder data for dataset {DATASET}.")

    def get_data(self):
        logging.info(f"Accessing encoder data for dataset {self.DATASET}.")
        return {
                "counts": self.encoder_counts,
                "stamps": self.encoder_stamps
        }

