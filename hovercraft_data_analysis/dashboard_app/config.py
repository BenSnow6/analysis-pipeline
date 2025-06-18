# dashboard_app/config.py

# Path to the main directory containing all experiment folders
# Adjust this path relative to where you run app.py, or use an absolute path.
DATA_REPO_PATH = '../../02_Evaluation_Experiments' # Updated path to point to the original experiments folder

# Subdirectory names expected within each experiment folder
GPS_SUBDIR = "GPS"
IMU_SUBDIR = "IMU"
ORIENTATIONS_FILENAME = "sensor_orientations.json"

# Default IMU sensor names/prefixes (if needed as a fallback)
IMU_SENSORS_DEFAULT = ["sensor_3", "sensor_4", "sensor_5", "sensor_wb", "sensor_wnb"]
IMU_MEASUREMENT_TYPES = ['accel', 'gyro', 'angle', 'mag']

# For mapping user-friendly sensor names to directory names if they differ
# (lowercase user-friendly name : actual directory name)
SENSOR_DIR_MAP = {
    "sensor_3": "Sensor_3",
    "sensor_4": "Sensor_4",
    "sensor_5": "Sensor_5",
    "sensor_wb": "Sensor_wb",
    "sensor_wnb": "Sensor_wnb",
} 