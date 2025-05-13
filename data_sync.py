import os
import numpy as np
import pandas as pd
from data_utils import list_experiment_types, list_experiments, load_experiment_data


def sync_data():
    """
    Reads each experiment, synchronizes IMU (accel+gyro+mag) and GPS on a uniform time_from_sync timeline
    based on a reference IMU stream (Sensor_3 accel preferred),
    writes stage1_sync/[type]/[run]/imu.parquet and gps.parquet,
    prints a one-line summary, and returns the last synced DataFrames.
    """
    total_interp = 0
    total_ffill = 0
    last_imu = None
    last_gps = None

    for exp_type in list_experiment_types():
        for run in list_experiments(exp_type):
            data = load_experiment_data(exp_type, run)
            if data is None:
                continue

            # Load GPS and index by raw 'time_from_sync'
            gps = data.get_gps_data()
            if gps is None or gps.empty:
                print(f"No GPS data for: {exp_type}/{run}, skipping")
                continue
            gps = gps.copy()
            if 'time_from_sync' not in gps.columns:
                print(f"No time_from_sync column in GPS for: {exp_type}/{run}, skipping")
                continue
            gps['time_from_sync'] = pd.to_numeric(gps['time_from_sync'], errors='coerce')
            gps = gps.dropna(subset=['time_from_sync']).set_index('time_from_sync').sort_index()

            # Choose reference IMU timeline: Sensor_3 accel preferred, else first accel
            ref_times = None
            for sensor in ['Sensor_3'] + [s for s in data.available_sensors if s != 'Sensor_3']:
                if 'accel' in data.available_imu_types.get(sensor, []):
                    df_ref = data.get_imu_data(sensor, 'accel')
                else:
                    continue
                if 'time_from_sync' not in df_ref.columns:
                    print(f"No time_from_sync in {sensor}/accel, skipping as reference")
                    continue
                ref_times = pd.to_numeric(df_ref['time_from_sync'], errors='coerce').dropna().values
                if len(ref_times) > 1:
                    break
            if ref_times is None or len(ref_times) < 2:
                print(f"No valid reference IMU times for: {exp_type}/{run}, skipping")
                continue

            # Collect IMU streams at that timeline
            imu_frames = []
            for sensor in data.available_sensors:
                for imu_type in ('accel', 'gyro', 'mag'):
                    if imu_type not in data.available_imu_types.get(sensor, []):
                        continue
                    df = data.get_imu_data(sensor, imu_type)
                    if df is None or df.empty:
                        continue
                    if 'time_from_sync' not in df.columns:
                        print(f"No time_from_sync column in {sensor}/{imu_type}, skipping")
                        continue
                    df = df.copy()
                    df['time_from_sync'] = pd.to_numeric(df['time_from_sync'], errors='coerce')
                    df = df.dropna(subset=['time_from_sync']).set_index('time_from_sync')
                    axes = data.get_sensor_info(imu_type)['axes']
                    cols = {axis: f"{sensor}_{axis}" for axis in axes if axis in df.columns}
                    if not cols:
                        continue
                    df = df[list(cols)].rename(columns=cols)
                    imu_frames.append(df)

            if not imu_frames:
                print(f"No IMU data for: {exp_type}/{run}, skipping")
                continue

            # Concatenate streams
            imu = pd.concat(imu_frames, axis=1).sort_index()

            # Resample: interpolate IMU, forward-fill GPS on reference times
            imu_sync = imu.reindex(ref_times).interpolate(method='linear')
            gps_sync = gps.reindex(ref_times).ffill()

            # Update counters
            total_interp += imu_sync.shape[0] - imu.shape[0]
            total_ffill += gps_sync.shape[0] - gps.shape[0]

            # Write results
            outdir = os.path.join('stage1_sync', exp_type, run)
            os.makedirs(outdir, exist_ok=True)
            imu_sync.to_parquet(os.path.join(outdir, 'imu.parquet'))
            gps_sync.to_parquet(os.path.join(outdir, 'gps.parquet'))

            last_imu, last_gps = imu_sync, gps_sync

    print(f"Interpolated {total_interp} IMU samples; forward-filled {total_ffill} GPS samples.")
    return last_imu, last_gps


if __name__ == '__main__':
    sync_data()
