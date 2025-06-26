# src/data_processing.py
import os
import pandas as pd
import numpy as np
from .classes import GPS, IMU  # Use relative import

def load_real_world_data(experiment_path):
    experiment_name = os.path.basename(experiment_path)
    gps_path = os.path.join(experiment_path, 'GPS', f"GPS_{experiment_name}.csv")
    gps = GPS(gps_path)
    gps.read_csv()
    gps.process_data()

    imu_dir = os.path.join(experiment_path, 'IMU')
    imu_sensors = ['3', '4', '5', 'wb', 'wnb']
    imu_objects = []
    for sensor in imu_sensors:
        sensor_path = os.path.join(imu_dir, f"Sensor_{sensor}")
        orientation_matrix = get_orientation_matrix(sensor)  # Define this function based on <Orientation example>
        imu = IMU(sensor_path, sensor, orientation_matrix, experiment_name)  # Pass experiment_name
        imu.read_csv_files()
        imu.apply_orientation()
        imu.process_data()
        imu_objects.append(imu)

    # Combine all derived data
    derived_data = gps.derived.copy()
    for imu in imu_objects:
        derived_data = derived_data.merge(imu.derived, on='time_from_sync', how='outer', suffixes=('', f'_{imu.sensor_name}'))

    return derived_data

def load_simulated_data(simulated_data_path):
    # Assuming simulated data is in CSV format with the same structure as derived_data
    simulated_data = pd.read_csv(simulated_data_path)
    return simulated_data

def join_data(real_data, simulated_data):
    combined_data = real_data.merge(simulated_data, on='time_from_sync', suffixes=('_real', '_sim'))
    return combined_data

def get_orientation_matrix(sensor_name):
    # Define orientation matrices based on <Orientation example>
    if sensor_name == 'wb':
        return np.array([[0, 0, 1],
                         [0, -1, 0],
                         [1, 0, 0]])
    elif sensor_name == '4':
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])
    elif sensor_name == '3':
        return np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]])
    # Add other sensors as needed
    else:
        return np.identity(3)
