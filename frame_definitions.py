import numpy as np
import json
from pathlib import Path

# --- Constants relevant for AHRS/Navigation ---

# Gravitational acceleration in NED frame (North, East, Down)
# Value is m/s^2
GRAVITY_NED = np.array([0.0, 0.0, 9.80665])

# Magnetic Declination for your test location
# Positive if Magnetic North is East of True North.
# Negative if Magnetic North is West of True North.
# Find this value for your specific location and date (e.g., from NOAA).
# Value in degrees.
MAGNETIC_DECLINATION_DEG = 0.6833  # Ryde, +0 deg 41 min (Positive East)

# --- Sensor Orientation Data ---

# Define the path to your sensor orientation JSON file
# Assuming it's in the same 'config' directory or provide an absolute/relative path
_current_dir = Path(__file__).resolve().parent
SENSOR_ORIENTATION_JSON_PATH = _current_dir / "sensor_orientations.json" # Or your specific path

# Hovercraft Body Frame Definition:
# X: Forward
# Y: Starboard (Right)
# Z: Downward

def _get_body_axis_vector(direction_str: str) -> np.ndarray:
    """
    Maps a direction string to a unit vector in the hovercraft body frame.
    Body Frame: X-Forward, Y-Starboard, Z-Downward.
    """
    direction_str = direction_str.lower()
    if direction_str == "forward":
        return np.array([1, 0, 0])
    elif direction_str == "aft" or direction_str == "backward":
        return np.array([-1, 0, 0])
    elif direction_str == "starboard" or direction_str == "right":
        return np.array([0, 1, 0])
    elif direction_str == "port" or direction_str == "left":
        return np.array([0, -1, 0])
    elif direction_str == "upward" or direction_str == "up":
        return np.array([0, 0, -1]) # Body Z is Down, so Upward is -Z_body
    elif direction_str == "downward" or direction_str == "down":
        return np.array([0, 0, 1])  # Body Z is Down, so Downward is +Z_body
    else:
        raise ValueError(f"Unknown direction string: {direction_str}")

def _create_R_bs_from_directions(x_dir_str: str, y_dir_str: str, z_dir_str: str) -> np.ndarray:
    """
    Creates the R_bs (Sensor to Body) DCM.
    The columns of R_bs are the sensor's X, Y, Z axes expressed in the body frame.
    v_body = R_bs @ v_sensor
    """
    e_x_sensor_in_body = _get_body_axis_vector(x_dir_str)
    e_y_sensor_in_body = _get_body_axis_vector(y_dir_str)
    e_z_sensor_in_body = _get_body_axis_vector(z_dir_str)

    # R_bs columns are [e_x_sensor_in_body, e_y_sensor_in_body, e_z_sensor_in_body]
    R_bs = np.column_stack([e_x_sensor_in_body, e_y_sensor_in_body, e_z_sensor_in_body])
    
    # Sanity check: R_bs should be a valid rotation matrix (orthonormal)
    if not np.allclose(R_bs @ R_bs.T, np.identity(3)):
        # This can happen if the provided directions are not orthogonal
        # e.g., x_direction="Forward", y_direction="Forward"
        print(f"Warning: Generated R_bs for directions ({x_dir_str}, {y_dir_str}, {z_dir_str}) is not orthonormal.")
        print(f"R_bs = \n{R_bs}")
        print(f"R_bs @ R_bs.T = \n{R_bs @ R_bs.T}")
        # Depending on severity, you might want to raise an error or handle it.
        # For now, we'll proceed but the user should check their JSON if this warning appears.

    return R_bs

SENSOR_TO_BODY_TRANSFORMS_DCM = {}

# Load sensor orientations from JSON and populate the DCM dictionary
try:
    with open(SENSOR_ORIENTATION_JSON_PATH, 'r') as f:
        sensor_orientations_data = json.load(f)

    # SENSOR_NAMES from data_utils.py for consistent naming if needed
    # SENSOR_NAMES_DATA_UTILS = ["Sensor_3", "Sensor_4", "Sensor_5", "Sensor_wb", "Sensor_wnb"]
    # We will map JSON device_name to data_utils SENSOR_NAMES style
    
    name_map = {
        "sensor_3": "Sensor_3",
        "sensor_4": "Sensor_4",
        "sensor_5": "Sensor_5",
        "sensor_wb": "Sensor_wb",
        "sensor_wnb": "Sensor_wnb",
        "gps": "GPS_Mount" # GPS mount orientation, may not be used for AHRS directly
    }

    for sensor_info in sensor_orientations_data:
        json_device_name = sensor_info.get("device_name")
        # Map to the capitalized version if it exists in our map, otherwise use as is or log warning
        # This allows flexibility if JSON names differ slightly but refer to the same conceptual sensor
        dcm_key_name = name_map.get(json_device_name, json_device_name)

        x_dir = sensor_info.get("x_direction")
        y_dir = sensor_info.get("y_direction")
        z_dir = sensor_info.get("z_direction")

        if not (dcm_key_name and x_dir and y_dir and z_dir):
            print(f"Warning: Incomplete data for an entry in sensor_orientations.json: {sensor_info}")
            continue
        
        R_bs = _create_R_bs_from_directions(x_dir, y_dir, z_dir)
        SENSOR_TO_BODY_TRANSFORMS_DCM[dcm_key_name] = R_bs
        print(f"Generated R_bs for {dcm_key_name}:\n{R_bs}")

except FileNotFoundError:
    print(f"Warning: Sensor orientation JSON file not found at {SENSOR_ORIENTATION_JSON_PATH}.")
    print("Falling back to default identity matrices for R_bs if any SENSOR_NAMES are predefined.")
    # Fallback to identity if file not found (as in previous version)
    # You might want to make this stricter depending on your workflow
    default_sensor_names = ["Sensor_3", "Sensor_4", "Sensor_5", "Sensor_wb", "Sensor_wnb"]
    for name in default_sensor_names:
        if name not in SENSOR_TO_BODY_TRANSFORMS_DCM:
            SENSOR_TO_BODY_TRANSFORMS_DCM[name] = np.identity(3)
            print(f"Warning: Using default identity R_bs for {name} due to missing JSON or entry.")
except json.JSONDecodeError:
    print(f"Warning: Error decoding JSON from {SENSOR_ORIENTATION_JSON_PATH}.")
    # Similar fallback or error handling
except Exception as e:
    print(f"An unexpected error occurred while loading sensor orientations: {e}")
    # Similar fallback

# Ensure all SENSOR_NAMES from data_utils have an entry, even if it's identity as a last resort
# This matches the SENSOR_NAMES list you have in data_utils.py
# If the JSON doesn't cover all of them, they'll get identity here.
# This part might be redundant if the JSON is expected to be complete.
_EXPECTED_IMU_KEYS = ["Sensor_3", "Sensor_4", "Sensor_5", "Sensor_wb", "Sensor_wnb"]
for key in _EXPECTED_IMU_KEYS:
    if key not in SENSOR_TO_BODY_TRANSFORMS_DCM:
        SENSOR_TO_BODY_TRANSFORMS_DCM[key] = np.identity(3)
        print(f"Warning: {key} not found in sensor_orientations.json or failed to parse. Using identity R_bs.")


def get_R_bs_dcm(sensor_name: str) -> np.ndarray:
    """
    Returns the Sensor-to-Body DCM for the given sensor name.
    v_body = R_bs @ v_sensor
    """
    if sensor_name not in SENSOR_TO_BODY_TRANSFORMS_DCM:
        # This case should be less likely now with the fallbacks, but good for robustness
        print(f"Warning: R_bs DCM for sensor '{sensor_name}' not found. Returning identity matrix.")
        return np.identity(3) 
    return SENSOR_TO_BODY_TRANSFORMS_DCM[sensor_name]

def get_R_sb_dcm(sensor_name: str) -> np.ndarray:
    """
    Returns the Body-to-Sensor DCM for the given sensor name.
    v_sensor = R_sb @ v_body
    """
    R_bs = get_R_bs_dcm(sensor_name)
    return R_bs.T # For DCMs, inverse is the transpose

# --- Example: Print out the loaded DCMs ---
if __name__ == "__main__":
    print("\n--- Loaded Sensor-to-Body DCMs (R_bs) ---")
    for name, dcm in SENSOR_TO_BODY_TRANSFORMS_DCM.items():
        print(f"\nSensor: {name}")
        print(dcm)
    
    print(f"\nExample for Sensor_3 (Body = R_bs @ Sensor):")
    print(get_R_bs_dcm("Sensor_3"))

    # Test with a vector: if sensor_3 x-axis is Upward (Body -Z), a reading of [1,0,0] on sensor_3
    # should correspond to [0,0,-1] in the body frame.
    v_sensor3_x_reading = np.array([1, 0, 0])
    v_body_from_s3_x = get_R_bs_dcm("Sensor_3") @ v_sensor3_x_reading
    print(f"A reading of [1,0,0] on Sensor_3 (X-axis) corresponds to Body vector: {v_body_from_s3_x}")