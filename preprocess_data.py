import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

# Assuming data_utils.py is in the same directory or accessible in PYTHONPATH
from data_utils import ExperimentData, load_experiment_data, SENSOR_NAMES, IMU_DF_COLUMN_MAP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MASTER_RATE_HZ = 100.0  # Desired common sampling rate in Hz
STATIC_DURATION_SEC = 5.0 # Duration at the start of data assumed to be static for gyro bias

# --- Step 1: Data Loading, Synchronisation & Resampling ---

def load_and_synchronize_data(
    exp_data: ExperimentData,
    target_sensor_name: str,
    master_rate_hz: float = MASTER_RATE_HZ
) -> Optional[pd.DataFrame]:
    """
    Loads IMU (accel, gyro, mag) and GPS data for a specific sensor,
    synchronizes them to a common time index, and resamples them.

    Args:
        exp_data: An ExperimentData object from data_utils.
        target_sensor_name: The name of the IMU sensor to process (e.g., "Sensor_3").
        master_rate_hz: The common sampling rate to resample data to.

    Returns:
        A Pandas DataFrame with synchronized and resampled data, or None if errors occur.
        The DataFrame will have a 'time_sec' index.
    """
    if target_sensor_name not in exp_data.available_sensors:
        logger.error(f"Target sensor {target_sensor_name} not available in experiment {exp_data.experiment_name}.")
        return None

    logger.info(f"Loading and synchronizing data for sensor: {target_sensor_name} at {master_rate_hz} Hz")

    # 1.1 Load individual dataframes
    imu_dfs = {}
    imu_types_to_load = ['accel', 'gyro', 'mag'] # Core IMU types for AHRS

    for imu_type in imu_types_to_load:
        if imu_type not in exp_data.available_imu_types[target_sensor_name]:
            logger.warning(f"{imu_type} data not available for {target_sensor_name}. Skipping.")
            continue
        df = exp_data.get_imu_data(target_sensor_name, imu_type)
        if df is None or df.empty:
            logger.warning(f"Could not load or empty data for {target_sensor_name}/{imu_type}.")
            continue
        
        # Get the name of the 'time_from_sync_...' column as it appears in the DataFrame 'df'
        # This name is the *value* associated with the key 'time_from_sync_raw' in IMU_DF_COLUMN_MAP.
        # For example, for 'accel', this will be 'time_from_sync_accel'.
        
        renamed_sync_column_in_df = IMU_DF_COLUMN_MAP[imu_type].get('time_from_sync_raw')

        if renamed_sync_column_in_df is None:
            # This case should ideally not happen if IMU_DF_COLUMN_MAP is correctly defined for all imu_types
            logger.error(f"Mapping for 'time_from_sync_raw' not found in IMU_DF_COLUMN_MAP for {imu_type}.")
            return None

        if renamed_sync_column_in_df not in df.columns:
            logger.error(f"Column '{renamed_sync_column_in_df}' (expected sync time column for {imu_type}) not found in the loaded DataFrame.")
            logger.debug(f"Available columns in df for {imu_type}: {df.columns.tolist()}") # Helpful for debugging
            return None
            
        df['time_sec'] = df[renamed_sync_column_in_df] # Use this correct column name
        df = df.set_index('time_sec')
        imu_dfs[imu_type] = df
        logger.debug(f"Loaded {target_sensor_name}/{imu_type} with {len(df)} rows. Index: {df.index.name}, Columns: {df.columns.tolist()}")


    if not all(imu_type in imu_dfs for imu_type in imu_types_to_load):
        logger.error(f"Missing one or more core IMU data types (accel, gyro, mag) for {target_sensor_name}.")
        return None

    gps_df = exp_data.get_gps_data()
    if gps_df is None or gps_df.empty:
        logger.warning("GPS data not loaded or is empty. Proceeding without GPS for now, but it will be needed later.")
        # Create an empty df with time_sec if needed for merging, or handle absence later
    else:
        if 'time_from_sync' not in gps_df.columns: # As per your _load_gps_data logic
            logger.error("'time_from_sync' column not found in GPS data.")
            # Attempt to use another timestamp if available or fail
            if 'Timestamp_dt' in gps_df.columns: # Assuming Timestamp_dt might be absolute
                 logger.warning("GPS 'time_from_sync' missing. Trying to infer from first IMU timestamp.")
                 if imu_dfs:
                     first_imu_time_col = IMU_DF_COLUMN_MAP[imu_types_to_load[0]]['t_raw']
                     first_imu_time_col_renamed = IMU_DF_COLUMN_MAP[imu_types_to_load[0]][first_imu_time_col]
                     if first_imu_time_col_renamed in imu_dfs[imu_types_to_load[0]].columns:
                         # This is a rough alignment and assumes GPS 'Timestamp' is absolute epoch
                         # And IMU 't_...' is also epoch. This part is tricky without clear common sync.
                         # For now, we rely on 'time_from_sync_...' being the primary mechanism.
                         pass # This path requires more robust handling if 'time_from_sync' is truly missing in GPS.
            # For now, let's assume 'time_from_sync' is present as per your data_utils.py
            # return None # Or handle gracefully if GPS sync time is critical here
        else:
            gps_df['time_sec'] = gps_df['time_from_sync']
            gps_df = gps_df.set_index('time_sec')
            logger.debug(f"Loaded GPS data with {len(gps_df)} rows. Index: {gps_df.index.name}")


    # 1.2 Determine common time range and create master time index
    all_indices = [df.index for df in imu_dfs.values()]
    if gps_df is not None and not gps_df.empty:
        all_indices.append(gps_df.index)

    if not all_indices:
        logger.error("No dataframes to process for common time range.")
        return None

    min_time = min(idx.min() for idx in all_indices if not idx.empty)
    max_time = max(idx.max() for idx in all_indices if not idx.empty)
    
    dt_master = 1.0 / master_rate_hz
    # Using np.arange for float steps; pd.to_timedelta might be an option if indices were datetime
    master_index_values = np.arange(min_time, max_time + dt_master, dt_master)
    master_index = pd.Index(master_index_values, name='time_sec')
    logger.info(f"Master time index created: {len(master_index)} points from {min_time:.3f}s to {max_time:.3f}s with dt={dt_master:.4f}s.")

    # 1.3 Resample and merge
    # Select and rename columns for merging
    # Accel: ax, ay, az
    # Gyro: gx, gy, gz (convert to rad/s later if in deg/s)
    # Mag: mx, my, mz
    
    df_accel = imu_dfs['accel'][[IMU_DF_COLUMN_MAP['accel']['x_raw'], 
                                 IMU_DF_COLUMN_MAP['accel']['y_raw'], 
                                 IMU_DF_COLUMN_MAP['accel']['z_raw']]].copy()
    df_accel.columns = ['ax', 'ay', 'az'] # Standardized names

    df_gyro = imu_dfs['gyro'][[IMU_DF_COLUMN_MAP['gyro']['x_raw'], 
                               IMU_DF_COLUMN_MAP['gyro']['y_raw'], 
                               IMU_DF_COLUMN_MAP['gyro']['z_raw']]].copy()
    df_gyro.columns = ['gx', 'gy', 'gz']

    df_mag = imu_dfs['mag'][[IMU_DF_COLUMN_MAP['mag']['x_raw'], 
                             IMU_DF_COLUMN_MAP['mag']['y_raw'], 
                             IMU_DF_COLUMN_MAP['mag']['z_raw']]].copy()
    df_mag.columns = ['mx', 'my', 'mz']

    # Resample IMU data
    accel_resampled = df_accel.reindex(master_index).interpolate(method='linear').ffill().bfill()
    gyro_resampled = df_gyro.reindex(master_index).interpolate(method='linear').ffill().bfill()
    mag_resampled = df_mag.reindex(master_index).interpolate(method='linear').ffill().bfill()
    
    # Combine IMU data
    synchronized_df = pd.concat([accel_resampled, gyro_resampled, mag_resampled], axis=1)

    # Resample and merge GPS data (if available)
    if gps_df is not None and not gps_df.empty:
        # Select relevant GPS columns (adjust names if different in your GPS files)
        # Common names: 'Latitude', 'Longitude', 'Altitude', 'Speed', 'Course'/'Track'
        gps_cols_to_keep = {
            'Lat': 'gps_lat', 'Lon': 'gps_lon', # From your example GPS_...csv
            'Altitude': 'gps_alt', 'Speed': 'gps_speed_knots', 'Course': 'gps_course_deg'
        }
        # Filter to columns that actually exist in gps_df
        actual_gps_cols = {k: v for k, v in gps_cols_to_keep.items() if k in gps_df.columns}
        if not actual_gps_cols:
             logger.warning("No standard GPS columns (Lat, Lon, etc.) found in GPS data. Skipping GPS merge.")
        else:
            gps_subset = gps_df[list(actual_gps_cols.keys())].copy()
            gps_subset.rename(columns=actual_gps_cols, inplace=True)
            gps_resampled = gps_subset.reindex(master_index).ffill().bfill()
            synchronized_df = pd.concat([synchronized_df, gps_resampled], axis=1)
            logger.info("GPS data resampled and merged.")
    else:
        logger.info("No GPS data to merge.")
        
    # Check for NaNs after all operations
    if synchronized_df.isnull().values.any():
        nan_cols = synchronized_df.columns[synchronized_df.isnull().any()].tolist()
        logger.warning(f"NaN values found in synchronized_df in columns: {nan_cols}. Consider data quality or interpolation limits.")
        # synchronized_df.fillna(0, inplace=True) # Or a more sophisticated fill

    logger.info(f"Data synchronized. Resulting DataFrame shape: {synchronized_df.shape}")
    return synchronized_df


# --- Step 2: Simplified Sensor Calibration ---

def basic_gyro_calibration(
    df: pd.DataFrame, 
    static_duration_sec: float = STATIC_DURATION_SEC
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Performs basic gyroscope calibration by removing bias estimated from an initial static period.
    Assumes gx, gy, gz are in deg/s and converts them to rad/s.

    Args:
        df: DataFrame with 'gx', 'gy', 'gz' columns (expected in deg/s).
        static_duration_sec: Duration from the start of the data to consider static.

    Returns:
        Tuple of:
            - DataFrame with added 'gx_cal_rps', 'gy_cal_rps', 'gz_cal_rps' (rad/s).
            - Estimated gyro bias vector [bias_x, bias_y, bias_z] (in original units, e.g. deg/s).
    """
    logger.info(f"Performing basic gyro calibration. Static duration: {static_duration_sec}s")
    
    # Ensure gyro columns exist
    if not all(col in df.columns for col in ['gx', 'gy', 'gz']):
        logger.error("Gyro columns (gx, gy, gz) not found for calibration.")
        raise ValueError("Missing gyro columns for calibration")

    # Determine static period from the start of the DataFrame's time index
    static_end_time = df.index.min() + static_duration_sec
    static_data = df[df.index <= static_end_time]

    if static_data.empty:
        logger.warning("No static data found for gyro bias estimation. Using zero bias.")
        gyro_bias = np.array([0.0, 0.0, 0.0])
    else:
        gyro_bias = static_data[['gx', 'gy', 'gz']].mean().values
        logger.info(f"Estimated gyro bias (deg/s): {gyro_bias}")

    # Subtract bias and convert to rad/s
    # Gyro data from SENSOR_INFO is in °/s, AHRS libraries usually expect rad/s
    df['gx_cal_rps'] = np.deg2rad(df['gx'] - gyro_bias[0])
    df['gy_cal_rps'] = np.deg2rad(df['gy'] - gyro_bias[1])
    df['gz_cal_rps'] = np.deg2rad(df['gz'] - gyro_bias[2])
    
    logger.info("Gyro data calibrated (bias removed) and converted to rad/s.")
    return df, gyro_bias


def check_accelerometer_static(df: pd.DataFrame, static_duration_sec: float = STATIC_DURATION_SEC):
    """
    Checks accelerometer readings during an initial static period.
    Assumes ax, ay, az are in m/s^2.
    """
    logger.info(f"Checking accelerometer static readings. Static duration: {static_duration_sec}s")
    
    if not all(col in df.columns for col in ['ax', 'ay', 'az']):
        logger.error("Accelerometer columns (ax, ay, az) not found for static check.")
        raise ValueError("Missing accelerometer columns for static check")

    static_end_time = df.index.min() + static_duration_sec
    static_data_accel = df[df.index <= static_end_time][['ax', 'ay', 'az']]

    if static_data_accel.empty:
        logger.warning("No static data found for accelerometer check.")
    else:
        mean_static_accel = static_data_accel.mean().values
        magnitude_static_accel = np.linalg.norm(mean_static_accel)
        logger.info(f"Mean static accelerometer vector [m/s^2]: {mean_static_accel}")
        logger.info(f"Magnitude of mean static acceleration: {magnitude_static_accel:.3f} m/s^2 (expected ~9.81)")

# --- Main processing example function ---
def process_experiment_data_for_heading(
    experiment_type: str,
    experiment_name: str,
    target_sensor_name: str,
    static_duration_sec: float = STATIC_DURATION_SEC
) -> Optional[pd.DataFrame]:
    """
    Main pipeline function to load, synchronize, and calibrate data for one experiment.
    """
    logger.info(f"Starting processing for {experiment_type}/{experiment_name}, sensor {target_sensor_name}")
    
    # Load experiment data structure
    exp_data_obj = load_experiment_data(experiment_type, experiment_name)
    if not exp_data_obj:
        logger.error("Failed to load experiment data object.")
        return None

    # Step 1: Load and Synchronize
    synced_df = load_and_synchronize_data(exp_data_obj, target_sensor_name, MASTER_RATE_HZ)
    if synced_df is None or synced_df.empty:
        logger.error("Failed to load and synchronize data.")
        return None
    
    # Step 2: Basic Calibrations
    # Gyro
    try:
        synced_df, gyro_bias = basic_gyro_calibration(synced_df, static_duration_sec)
    except ValueError as e:
        logger.error(f"Error in gyro calibration: {e}")
        return None # Or handle differently if gyro cal is optional

    # Accel check (no modification, just logging)
    try:
        check_accelerometer_static(synced_df, static_duration_sec)
    except ValueError as e:
        logger.error(f"Error in accel check: {e}")
        # Continue even if accel check fails, as it's informational here

    # Magnetometer check (will be done visually or when using it in AHRS)
    # For now, we assume 'mx', 'my', 'mz' are usable as is.

    logger.info("Preprocessing and basic calibration complete.")
    return synced_df


if __name__ == '__main__':
    # --- Example Usage ---
    # Replace with your actual experiment type and name
    TEST_EXPERIMENT_TYPE = "1a_1_Minimum_Radius_Turn" 
    TEST_EXPERIMENT_NAME = "007_Fast_stbd_turn_1" # Make sure this experiment exists
    TARGET_IMU_SENSOR = "Sensor_3" # Choose one of your SENSOR_NAMES

    # Check if default data paths exist (as in your data_utils.py example)
    # This section is for making the example runnable if data isn't in standard paths
    from pathlib import Path
    from data_utils import DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE
    actual_roots_exist = any(Path(p).exists() for p in DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE)
    current_test_roots = DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE

    if not actual_roots_exist and Path.cwd().name != "01_analysis_pipeline":
        print("\n!!! WARNING: Default data paths not found. Will try to use/create a dummy structure for demonstration. !!!")
        print("!!! This example will be more meaningful if run from the root of your project, or if paths are adjusted. !!!")
        
        # Create dummy data if it doesn't exist, mirroring data_utils example
        TEST_ROOT_PATH = Path("test_data_root/analysis-pipeline/02_Evaluation_Experiments")
        if not (TEST_ROOT_PATH / TEST_EXPERIMENT_TYPE / "afternoon" / TEST_EXPERIMENT_NAME).exists():
            TEST_ROOT_PATH.mkdir(parents=True, exist_ok=True)
            exp_type_path_dummy = TEST_ROOT_PATH / TEST_EXPERIMENT_TYPE / "afternoon" / TEST_EXPERIMENT_NAME
            exp_type_path_dummy.mkdir(parents=True, exist_ok=True)
            
            (exp_type_path_dummy / "GPS").mkdir(exist_ok=True)
            # GPS with time_from_sync
            (exp_type_path_dummy / "GPS" / f"GPS_{TEST_EXPERIMENT_NAME}.csv").write_text(
                "Timestamp,Lat,Lon,Speed,Course,time_from_sync\n"
                "1678886400,50.1,-1.0,10,45,0.0\n"
                "1678886401,50.2,-1.1,10.1,45.1,1.0\n"
                "1678886402,50.2,-1.1,10.2,45.2,2.0\n"
                "1678886403,50.2,-1.1,10.3,45.3,3.0\n"
                "1678886404,50.2,-1.1,10.4,45.4,4.0\n"
                "1678886405,50.2,-1.1,10.5,45.5,5.0\n"
                "1678886406,50.2,-1.1,10.6,45.6,6.0\n"
            )
            
            sensor_path_dummy = exp_type_path_dummy / "IMU" / TARGET_IMU_SENSOR
            sensor_path_dummy.mkdir(parents=True, exist_ok=True)
            # t_raw, x, y, z, time_from_sync_raw
            (sensor_path_dummy / f"accel_{TEST_EXPERIMENT_NAME}.csv").write_text(
                "1.0,0.1,0.2,9.8,0.001\n1.01,0.1,0.2,9.81,0.011\n1.02,0.1,0.2,9.82,0.021\n"
                "1.03,0.1,0.2,9.83,0.031\n1.04,0.1,0.2,9.84,0.041\n1.05,0.1,0.2,9.85,0.051\n" # More data for 100Hz
                + "\n".join([f"{1.05 + i*0.01},0.1,0.2,{9.85+i*0.001},{0.051+i*0.01}" for i in range(1, 600)])
            )
            (sensor_path_dummy / f"gyro_{TEST_EXPERIMENT_NAME}.csv").write_text(
                "1.0,0.5,0.2,0.3,0.001\n1.01,0.51,0.21,0.31,0.011\n1.02,0.52,0.22,0.32,0.021\n"
                "1.03,0.53,0.23,0.33,0.031\n1.04,0.54,0.24,0.34,0.041\n1.05,0.55,0.25,0.35,0.051\n"
                + "\n".join([f"{1.05 + i*0.01},{0.55+i*0.001},{0.25+i*0.001},{0.35+i*0.001},{0.051+i*0.01}" for i in range(1, 600)])
            )
            (sensor_path_dummy / f"mag_{TEST_EXPERIMENT_NAME}.csv").write_text(
                "1.0,20.0,5.0,45.0,0.001\n1.01,20.1,5.1,45.1,0.011\n1.02,20.2,5.2,45.2,0.021\n"
                "1.03,20.3,5.3,45.3,0.031\n1.04,20.4,5.4,45.4,0.041\n1.05,20.5,5.5,45.5,0.051\n"
                + "\n".join([f"{1.05 + i*0.01},{20.5+i*0.01},{5.5+i*0.01},{45.5+i*0.01},{0.051+i*0.01}" for i in range(1, 600)])
            )
            print(f"Created dummy data at {TEST_ROOT_PATH}")
            current_test_roots = [str(TEST_ROOT_PATH)] # Override default roots for the test
            # Crucially, update data_utils.DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE for this run
            import data_utils
            data_utils.DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE = current_test_roots


    # Process the data
    processed_dataframe = process_experiment_data_for_heading(
        TEST_EXPERIMENT_TYPE,
        TEST_EXPERIMENT_NAME,
        TARGET_IMU_SENSOR
    )

    if processed_dataframe is not None:
        print("\n--- Processed DataFrame Head ---")
        print(processed_dataframe.head())
        print("\n--- Processed DataFrame Info ---")
        processed_dataframe.info()
        print("\n--- Processed DataFrame Description ---")
        print(processed_dataframe.describe())

        # Example of checking a specific column
        if 'gx_cal_rps' in processed_dataframe.columns:
            print(f"\nGyro X (calibrated, rad/s) mean: {processed_dataframe['gx_cal_rps'].mean():.4f}")
        
        # Clean up dummy structure if it was created for this test run
        if not actual_roots_exist and Path("test_data_root").exists():
            import shutil
            # shutil.rmtree("test_data_root")
            print("\nCleaned up dummy test_data_root directory (or left for inspection).")
    else:
        print("Data processing failed.")