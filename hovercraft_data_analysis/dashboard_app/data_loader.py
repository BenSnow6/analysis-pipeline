# dashboard_app/data_loader.py
import os
import json
import pandas as pd
import config # Changed from relative import

def get_experiment_folders(data_repo_path):
    """Scans the data_repo_path recursively for valid experiment folders."""
    print(f"Debug: Scanning for experiments in: {data_repo_path}") # Added Debug print
    experiments = {}
    if not os.path.isdir(data_repo_path):
        print(f"Error: Data repository path not found: {data_repo_path}")
        return experiments

    # os.walk goes through the directory tree
    for root, dirs, files in os.walk(data_repo_path):
        # Check if the current directory ('root') contains both GPS and IMU subdirectories
        # Use os.path.join to be safe, although checking 'in dirs' might be okay if names are exact
        has_gps = os.path.isdir(os.path.join(root, config.GPS_SUBDIR))
        has_imu = os.path.isdir(os.path.join(root, config.IMU_SUBDIR))

        if has_gps and has_imu:
            # This directory is a valid experiment run folder
            experiment_path = root
            # Create a user-friendly name from the relative path
            relative_path = os.path.relpath(experiment_path, data_repo_path)
            # Use forward slashes for display consistency across OS
            display_name = relative_path.replace(os.sep, '/')
            print(f"Debug: Found potential experiment: {display_name} at {experiment_path}") # Added Debug print
            experiments[display_name] = experiment_path # Store display name and full path

            # Optional: Prevent os.walk from descending further into GPS/IMU folders
            # This avoids finding nested experiments if they accidentally exist and speeds up scan
            dirs[:] = [d for d in dirs if d not in [config.GPS_SUBDIR, config.IMU_SUBDIR]]

    # Sort experiments by name for a consistent dropdown order
    sorted_experiments = dict(sorted(experiments.items()))
    return sorted_experiments

def find_file_in_experiment(experiment_base_path, data_type_subdir, file_prefix_pattern, specific_tag=None):
    """
    Finds a data file within a specific subdirectory of an experiment.
    Example: experiment_base_path = /path/to/experiment_007_fast_turn
             data_type_subdir = "GPS" or "IMU/Sensor_3"
             file_prefix_pattern = "GPS_*.csv" or "accel_*.csv"
             specific_tag: e.g., "007_Fast_stbd_turn_1" (optional, tries to match if provided)
    """
    search_dir = os.path.join(experiment_base_path, data_type_subdir)
    print(f"Debug FindFile: Searching in '{search_dir}' for pattern '{file_prefix_pattern}' with tag '{specific_tag}'")
    if not os.path.isdir(search_dir):
        print(f"Warning: Data directory not found: {search_dir}")
        return None

    candidate_files = []
    files_in_dir = os.listdir(search_dir)
    print(f"Debug FindFile: Files found in dir: {files_in_dir}")
    for fname in files_in_dir:
        # Check both start and end, allowing for variable tags in the middle
        prefix = file_prefix_pattern.split('*')[0]
        suffix = file_prefix_pattern.split('*')[-1] if '*' in file_prefix_pattern else ''
        
        is_match = fname.startswith(prefix)
        if suffix:
            is_match = is_match and fname.endswith(suffix)

        print(f"Debug FindFile: Checking '{fname}'. StartsWith('{prefix}'): {fname.startswith(prefix)}. EndsWith('{suffix}'): {fname.endswith(suffix) if suffix else 'N/A'}. Overall match: {is_match}")

        if is_match:
            print(f"Debug FindFile: '{fname}' matches pattern. Checking tag '{specific_tag}'. Tag in fname: {specific_tag in fname if specific_tag else 'N/A'}")
            if specific_tag and specific_tag in fname:
                print(f"Debug FindFile: Tag found! Returning exact match: {os.path.join(search_dir, fname)}")
                return os.path.join(search_dir, fname) # Prioritize file with specific tag
            candidate_files.append(fname)

    if candidate_files:
        # If no specific tag match or multiple candidates, return the first one found.
        # You might want more sophisticated logic here (e.g., alphabetical sort, check timestamps)
        # For now, print a warning if multiple files are found without a specific tag match.
        if len(candidate_files) > 1 and not specific_tag:
             print(f"Warning: Multiple files found matching '{file_prefix_pattern}' in {search_dir}: {candidate_files}. Using: {candidate_files[0]}")
        elif specific_tag: # if specific_tag was provided but no exact match, still use the first candidate
            print(f"Warning: Specific tag '{specific_tag}' not found in any files matching '{file_prefix_pattern}' in {search_dir}. Using first candidate: {candidate_files[0]}")
        #else: # Only one candidate found, use it silently.
        #    print(f"Found candidate in {search_dir} for {file_prefix_pattern}: {candidate_files}. Using: {candidate_files[0]}")
        return os.path.join(search_dir, candidate_files[0])

    print(f"Warning: No file matching '{file_prefix_pattern}' found in {search_dir}")
    return None


def load_sensor_orientations(experiment_path):
    """Loads sensor orientations from sensor_orientations.json for a given experiment."""
    # Look for sensor_orientations.json in the *parent* of the specific run folder first
    # Assumes the category/timeslot level might hold it, or the root experiment folder.
    potential_paths = [
        os.path.join(os.path.dirname(experiment_path), config.ORIENTATIONS_FILENAME),
        os.path.join(experiment_path, config.ORIENTATIONS_FILENAME) # Fallback to looking inside the run folder itself
        # Add more levels up if needed, e.g., os.path.dirname(os.path.dirname(experiment_path))
    ]
    
    orientations_file_path = None
    for path in potential_paths:
        if os.path.exists(path):
            orientations_file_path = path
            break # Use the first one found

    default_orientations = {}
    if not orientations_file_path:
         # Also check the root data repo path specified in config as a last resort
        root_orientation_path = os.path.join(config.DATA_REPO_PATH, config.ORIENTATIONS_FILENAME)
        if os.path.exists(root_orientation_path):
             orientations_file_path = root_orientation_path
        else:
             print(f"Warning: '{config.ORIENTATIONS_FILENAME}' not found in {experiment_path}, its parent, or the root data directory ({config.DATA_REPO_PATH}).")
             return default_orientations

    try:
        with open(orientations_file_path, 'r') as f:
            orientations_list = json.load(f)
            # Ensure keys are lowercase for consistent lookup
            return {str(item['device_name']).lower(): item for item in orientations_list}
    except FileNotFoundError: # Should not happen due to os.path.exists check, but good practice
        print(f"Warning: '{config.ORIENTATIONS_FILENAME}' check passed but file not found at {orientations_file_path} during open.")
    except json.JSONDecodeError:
        print(f"Warning: Error decoding '{config.ORIENTATIONS_FILENAME}' from {orientations_file_path}.")
    except KeyError as e:
        print(f"Warning: Missing key {e} in item within '{config.ORIENTATIONS_FILENAME}' from {orientations_file_path}.")
    except Exception as e:
        print(f"Error loading sensor orientations from {orientations_file_path}: {e}")
    return default_orientations


def load_gps_data(experiment_path, experiment_file_tag=None):
    """Loads GPS data for the selected experiment."""
    if not experiment_file_tag:
        experiment_file_tag = os.path.basename(experiment_path)

    gps_file_relative = find_file_in_experiment(experiment_path, config.GPS_SUBDIR, "GPS_*.csv", specific_tag=experiment_file_tag)

    if gps_file_relative:
        gps_file_absolute = os.path.abspath(gps_file_relative)
        # Optional: Add debug print for path resolution if needed
        # print(f"Debug GPS Load: Resolved relative path '{gps_file_relative}' to absolute path '{gps_file_absolute}'")
        if os.path.exists(gps_file_absolute):
            try:
                df = pd.read_csv(gps_file_absolute) # Load using absolute path
                if 'time_from_sync' in df.columns:
                    df['time_from_sync'] = pd.to_numeric(df['time_from_sync'], errors='coerce')
                    df.dropna(subset=['time_from_sync'], inplace=True)
                    return df.sort_values('time_from_sync')
                elif 'Time' in df.columns:
                    try:
                        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                    except Exception as time_parse_error:
                         print(f"Warning: Could not parse 'Time' column in {gps_file_absolute} as datetime: {time_parse_error}. Skipping time_from_sync calculation from 'Time'.")
                         df['Time'] = pd.NaT
                    df.dropna(subset=['Time'], inplace=True)
                    if not df.empty:
                        df['time_from_sync'] = (df['Time'] - df['Time'].min()).dt.total_seconds()
                        return df.sort_values('time_from_sync')
                    else:
                         print(f"Warning: GPS data in {gps_file_absolute} became empty after dropping rows with unparseable 'Time'.")
                         return pd.DataFrame()
                print(f"Warning: Usable time column ('time_from_sync' or 'Time' with parseable values) not found in GPS data {gps_file_absolute}.")
                return df
            except Exception as e:
                print(f"Error loading or processing GPS data from {gps_file_absolute}: {e}")
                return pd.DataFrame()
        # else: # Optional: Add debug print if absolute path check fails
        #     print(f"Debug GPS Load: Absolute path check failed for {gps_file_absolute}")

    # If gps_file_relative was None or os.path.exists failed:
    # ... (existing debug logic for not found) ...
    # if not gps_file_relative:
    #     print(f"Debug: GPS file not found for experiment '{os.path.basename(experiment_path)}' ...")

    return pd.DataFrame()


def load_imu_data(experiment_path, sensor_name_user, measurement_type, experiment_file_tag=None):
    """Loads IMU data for the selected experiment, sensor, and measurement type."""
    if not experiment_file_tag:
         experiment_file_tag = os.path.basename(experiment_path)

    actual_sensor_dir_name = config.SENSOR_DIR_MAP.get(sensor_name_user.lower())
    if not actual_sensor_dir_name:
        print(f"Warning: Sensor directory mapping not found for '{sensor_name_user}' ...")
        actual_sensor_dir_name = sensor_name_user

    imu_sensor_subdir = os.path.join(config.IMU_SUBDIR, actual_sensor_dir_name)
    file_pattern = f"{measurement_type}_*.csv"
    imu_file_relative = find_file_in_experiment(experiment_path, imu_sensor_subdir, file_pattern, specific_tag=experiment_file_tag)

    if imu_file_relative:
        imu_file_absolute = os.path.abspath(imu_file_relative)
        # Optional: Add debug print for path resolution if needed
        # print(f"Debug IMU Load: Resolved relative path '{imu_file_relative}' to absolute path '{imu_file_absolute}'")
        if os.path.exists(imu_file_absolute):
            try:
                df = pd.read_csv(imu_file_absolute) # Load using absolute path
                # Debug prints remain for now
                print(f"Debug IMU Load: File {os.path.basename(imu_file_absolute)} loaded. Shape: {df.shape}. Dtypes:\\n{df.dtypes}")
                if not df.empty:
                    print(f"Debug IMU Load: Head:\\n{df.head()}")

                if 'time_from_sync' in df.columns:
                    original_len = len(df)
                    numeric_time_col = pd.to_numeric(df['time_from_sync'], errors='coerce')
                    is_valid_time = numeric_time_col.notna()
                    num_valid = is_valid_time.sum()
                    num_invalid = original_len - num_valid
                    if num_invalid == original_len:
                        print(f"ERROR: All {original_len} rows in {imu_file_absolute} have non-numeric 'time_from_sync' values...")
                        return pd.DataFrame()
                    if num_invalid > 0:
                        print(f"Warning: Dropping {num_invalid} out of {original_len} rows from {imu_file_absolute} ...")
                        df = df[is_valid_time].copy()
                        df['time_from_sync'] = numeric_time_col[is_valid_time]
                    else:
                        df['time_from_sync'] = numeric_time_col
                    return df.sort_values('time_from_sync')
                else:
                    print(f"Warning: 'time_from_sync' column not found in IMU data {imu_file_absolute}. Cannot plot against time.")
                    return df
            except Exception as e:
                print(f"Error loading or processing IMU data from {imu_file_absolute}: {e}")
                return pd.DataFrame()
        # else: # Optional: Add debug print if absolute path check fails
        #     print(f"Debug IMU Load: Absolute path check failed for {imu_file_absolute}")

    # If imu_file_relative was None or os.path.exists failed:
    # ... (existing debug logic for not found) ...
    # if not imu_file_relative:
    #     print(f"Debug: IMU file not found for experiment '{os.path.basename(experiment_path)}' ...")

    return pd.DataFrame()


def get_imu_sensors_for_experiment(experiment_path, orientations):
    """Gets a list of IMU sensors available for the given experiment."""
    # Priority 1: Get from orientations file if loaded successfully
    if orientations:
        # Filter keys from orientations based on whether they exist in SENSOR_DIR_MAP
        # Ensures we only list sensors the rest of the code knows how to find directories for.
        known_orientation_sensors = [key for key in orientations.keys()
                                     if key != 'gps' and key.lower() in config.SENSOR_DIR_MAP]
        if known_orientation_sensors:
            #print(f"Debug: Found sensors in orientations file: {known_orientation_sensors}")
            return sorted(known_orientation_sensors) # Sort for consistent order

    # Priority 2: Fallback - check subdirectories in the IMU folder of the specific experiment run
    imu_exp_dir = os.path.join(experiment_path, config.IMU_SUBDIR)
    found_sensors = []
    if os.path.isdir(imu_exp_dir):
        for item in os.listdir(imu_exp_dir):
            item_path = os.path.join(imu_exp_dir, item)
            # Check if 'item' (directory name) corresponds to a known sensor key in SENSOR_DIR_MAP
            for user_friendly_key, actual_dir_name in config.SENSOR_DIR_MAP.items():
                if actual_dir_name == item and os.path.isdir(item_path):
                    found_sensors.append(user_friendly_key) # Add the user-friendly key
                    break # Move to the next item in listdir
    if found_sensors:
        #print(f"Debug: Found sensors by scanning IMU directory: {found_sensors}")
        return sorted(found_sensors) # Sort for consistent order

    # Priority 3: Absolute fallback - return default list from config
    print(f"Warning: Could not determine IMU sensors from orientations file or by scanning directory {imu_exp_dir}. Falling back to default list from config.")
    return config.IMU_SENSORS_DEFAULT 