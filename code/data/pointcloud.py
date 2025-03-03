import numpy as np
import scipy.io as sio
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class PointCloud:
    def __init__(self, PATH: str, MODEL_NAME: str, PC_NUM: int):
        self.pose = np.eye(4)
        self.target_pc = []
        logging.info(f"Initializing PointCloud with model '{MODEL_NAME}' and pointcloud number {PC_NUM}")
        logging.info(f"Attempting to load canonical point cloud data for model '{MODEL_NAME}'.")
        model_fname = os.path.join(PATH, MODEL_NAME, 'model.mat')
        model = sio.loadmat(model_fname)
        self.source_pc = model['Mdata'].T / 1000.0
        logging.info(f"Successfully loaded canonical point cloud data for model '{MODEL_NAME}'.")
        logging.info(f"Attempting to load point cloud data for PC number {PC_NUM}.")
        
        for theNum in range(PC_NUM):
            pc_fname = os.path.join(PATH, MODEL_NAME, f'{theNum}.npy')
            pc_data = np.load(pc_fname)  # Shape: (N, 3)
            self.target_pc.append(pc_data) 
     
        logging.info(f"Successfully loaded point cloud data for PC number {PC_NUM}.")

    def get_data(self):
        logging.info(f"Accessing point cloud data for source and target")
        return {
                "source": self.source_pc,
                "target": self.target_pc
        }
