# src/classes.py
import os
import pandas as pd
import numpy as np

class GPS:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.derived = None

    def read_csv(self):
        self.data = pd.read_csv(self.filepath)

    def process_data(self):
        # Example: Extract relevant columns for derived data
        self.derived = self.data[['time_from_sync', 'SpeedKPH']].copy()
        # Add more derived calculations as needed

class IMU:
    def __init__(self, sensor_path, sensor_name, orientation_matrix, experiment_name):
        self.sensor_path = sensor_path
        self.sensor_name = sensor_name
        self.orientation_matrix = orientation_matrix
        self.experiment_name = experiment_name
        self.accel = None
        self.angle = None
        self.gyro = None
        self.mag = None
        self.derived = None

    def read_csv_files(self):
        # Construct file names using experiment_name
        self.accel = pd.read_csv(os.path.join(self.sensor_path, f"accel_{self.experiment_name}.csv"))
        self.angle = pd.read_csv(os.path.join(self.sensor_path, f"angle_{self.experiment_name}.csv"))
        self.gyro = pd.read_csv(os.path.join(self.sensor_path, f"gyro_{self.experiment_name}.csv"))
        self.mag = pd.read_csv(os.path.join(self.sensor_path, f"mag_{self.experiment_name}.csv"))

    def apply_orientation(self):
        # Apply the orientation matrix to transform sensor data to body frame
        # Placeholder for actual transformation logic
        # Example: Rotate accelerometer data
        accel_values = self.accel[['x', 'y', 'z']].values
        rotated_accel = accel_values.dot(self.orientation_matrix.T)
        self.accel[['x', 'y', 'z']] = rotated_accel

        # Similarly, apply to other sensor data if needed

    def process_data(self):
        # Example: Calculate pitch, roll, yaw from angle data
        self.derived = self.angle[['time_from_sync', 'x', 'y', 'z']].copy()
        self.derived.rename(columns={'x': f'pitch_{self.sensor_name}', 
                                     'y': f'roll_{self.sensor_name}', 
                                     'z': f'yaw_{self.sensor_name}'}, inplace=True)
        # Add more derived calculations as needed
