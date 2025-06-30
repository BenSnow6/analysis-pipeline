# dashboard_app/config.py

from pathlib import Path

# Import from centralized path configuration
from src.config.paths import DATA_DIR, MORNING_DATA_DIR, AFTERNOON_DATA_DIR

# Path to the main directory containing all experiment folders
# This now uses the centralized path configuration
DATA_REPO_PATH = DATA_DIR  # Points to the data directory containing morning/afternoon

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