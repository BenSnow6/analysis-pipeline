import os
import glob
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---

# Sensor information for plotting and interpretation
SENSOR_INFO = {
    'accel': {
        'unit': 'm/s²',
        'ylabel': 'Acceleration [m/s²]',
        'axes': ['ax', 'ay', 'az']
    },
    'gyro': {
        'unit': '°/s',
        'ylabel': 'Angular Velocity [°/s]',
        'axes': ['gx', 'gy', 'gz']
    },
    'mag': {
        'unit': 'µT',
        'ylabel': 'Magnetic Field [µT]',
        'axes': ['mx', 'my', 'mz']
    },
    'angle': {
        'unit': '°',
        'ylabel': 'Angle [°]',
        'axes': ['roll', 'pitch', 'yaw'] # Assuming x, y, z correspond to roll, pitch, yaw
    },
    'quat': { # Assuming quat files also have x,y,z named columns from the generic spec
        'unit': '', # Unitless
        'ylabel': 'Quaternion Component',
        'axes': ['qx', 'qy', 'qz'] # If it's w,x,y,z use ['qw', 'qx', 'qy', 'qz']
                                  # Based on "t,x,y,z,time_from_sync" for all IMU data
    }
}

# Column names as specified by the user for IMU data files
# "Each IMU data file contains the following columns:
# accel: t,x,y,z,time_from_sync
# angle: t,x,y,z,time_from_sync
# gyro: t,x,y,z,time_from_sync
# mag: t,x,y,z,time_from_sync"
# Assuming 'quat' follows the same t,x,y,z,time_from_sync structure if present.
IMU_RAW_COLUMNS = ['t_raw', 'x_raw', 'y_raw', 'z_raw', 'time_from_sync_raw']

# More descriptive column names for use in DataFrames
# We will map the raw columns to these during loading
IMU_DF_COLUMN_MAP = {
    'accel': {'t_raw': 't_accel', 'x_raw': 'ax', 'y_raw': 'ay', 'z_raw': 'az', 'time_from_sync_raw': 'time_from_sync_accel'},
    'angle': {'t_raw': 't_angle', 'x_raw': 'roll', 'y_raw': 'pitch', 'z_raw': 'yaw', 'time_from_sync_raw': 'time_from_sync_angle'},
    'gyro':  {'t_raw': 't_gyro', 'x_raw': 'gx', 'y_raw': 'gy', 'z_raw': 'gz', 'time_from_sync_raw': 'time_from_sync_gyro'},
    'mag':   {'t_raw': 't_mag', 'x_raw': 'mx', 'y_raw': 'my', 'z_raw': 'mz', 'time_from_sync_raw': 'time_from_sync_mag'},
    'quat':  {'t_raw': 't_quat', 'x_raw': 'qx', 'y_raw': 'qy', 'z_raw': 'qz', 'time_from_sync_raw': 'time_from_sync_quat'}
    # If quat is w,x,y,z, then this map and IMU_RAW_COLUMNS need adjustment for quat specifically.
    # For now, assuming 3 data components (x,y,z) + timestamp + sync_time for all.
}


IMU_DATA_TYPES = ["accel", "angle", "gyro", "mag", "quat"]
SENSOR_NAMES = ["Sensor_3", "Sensor_4", "Sensor_5", "Sensor_wb", "Sensor_wnb"]

# Dynamic path discovery - Enhanced path management
def get_default_root_paths() -> List[str]:
    """Dynamically find experiment base directories"""
    current_dir = Path.cwd()
    potential_paths = [
        current_dir / "02_Evaluation_Experiments",
        current_dir / "01_analysis_pipeline" / "02_Evaluation_Experiments",
        current_dir / "analysis-pipeline" / "02_Evaluation_Experiments"
    ]
    
    existing_paths = [str(p) for p in potential_paths if p.exists()]
    
    # Fallback to hardcoded paths if none found
    if not existing_paths:
        existing_paths = ["02_Evaluation_Experiments"]
    
    logger.info(f"Found experimental base paths: {existing_paths}")
    return existing_paths

# Default root paths (dynamically determined)
DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE = get_default_root_paths()

# --- ExperimentData Class ---
class ExperimentData:
    """
    Holds data and metadata for a single experiment with lazy loading capabilities.
    """
    def __init__(self, experiment_name: str, experiment_type: str,
                 base_path: Path, full_path: Path, time_of_day: Optional[str] = None):
        self.experiment_name: str = experiment_name
        self.experiment_type: str = experiment_type
        self.base_path: Path = base_path # e.g., .../02_Evaluation_Experiments
        self.full_path: Path = full_path # Full path to specific experiment folder
        self.time_of_day: Optional[str] = time_of_day

        self.gps_data: Optional[pd.DataFrame] = None
        # Lazy loading implementation - store file paths instead of data
        self.imu_data: Dict[str, Dict[str, pd.DataFrame]] = {sensor: {} for sensor in SENSOR_NAMES}
        self._imu_file_paths: Dict[str, Dict[str, Path]] = {sensor: {} for sensor in SENSOR_NAMES}
        self._loaded_imu: Dict[str, Dict[str, bool]] = {sensor: {} for sensor in SENSOR_NAMES}
        
        self.available_sensors: List[str] = []
        self.available_imu_types: Dict[str, List[str]] = {sensor: [] for sensor in SENSOR_NAMES}

        self.misc_files: Dict[str, Path] = {} # For other files like .json, .log, .html
        self.file_structure_type: Optional[str] = None  # Track detected structure type

    def __repr__(self) -> str:
        gps_loaded = "Yes" if self.gps_data is not None else "No"
        imu_sensors_count = len(self.available_sensors)
        return (f"<ExperimentData: {self.experiment_type}/{self.experiment_name} "
                f"(Time: {self.time_of_day if self.time_of_day else 'N/A'}), "
                f"GPS: {gps_loaded}, IMU Sensors: {imu_sensors_count}>")

    def get_gps_data(self) -> Optional[pd.DataFrame]:
        """Returns the GPS DataFrame."""
        return self.gps_data

    def get_imu_data(self, sensor_name: str, imu_type: str, reload: bool = False) -> Optional[pd.DataFrame]:
        """
        Returns a specific IMU DataFrame for a given sensor and data type with lazy loading.
        Example: exp_data.get_imu_data('Sensor_3', 'accel')
        
        Args:
            sensor_name: Name of the sensor
            imu_type: Type of IMU data (accel, gyro, etc.)
            reload: Force reload from file if True
        
        Returns:
            DataFrame if available, None otherwise
        """
        # Check if data exists and is loaded
        if (sensor_name in self.imu_data and 
            imu_type in self.imu_data[sensor_name] and 
            not reload and
            self._loaded_imu.get(sensor_name, {}).get(imu_type, False)):
            return self.imu_data[sensor_name][imu_type]
        
        # Load data if file path exists
        if (sensor_name in self._imu_file_paths and 
            imu_type in self._imu_file_paths[sensor_name]):
            file_path = self._imu_file_paths[sensor_name][imu_type]
            if file_path and file_path.is_file():
                try:
                    logger.debug(f"Lazy loading {sensor_name}/{imu_type} from {file_path}")
                    df_imu = self._load_imu_file(file_path, imu_type)
                    self.imu_data[sensor_name][imu_type] = df_imu
                    self._loaded_imu[sensor_name][imu_type] = True
                    return df_imu
                except Exception as e:
                    logger.error(f"Failed to load IMU data from {file_path}: {e}")
                    return None
        
        logger.warning(f"IMU data for {sensor_name}/{imu_type} not found or not available.")
        return None

    def _load_imu_file(self, file_path: Path, imu_type: str) -> pd.DataFrame:
        """Load and process an IMU file"""
        df_imu = pd.read_csv(file_path, header=None, names=IMU_RAW_COLUMNS, on_bad_lines='warn')
        
        column_map = IMU_DF_COLUMN_MAP[imu_type]
        df_imu.rename(columns=column_map, inplace=True)
        
        time_sync_col_renamed = column_map.get('time_from_sync_raw')
        if time_sync_col_renamed and time_sync_col_renamed in df_imu.columns:
            df_imu[time_sync_col_renamed] = pd.to_numeric(df_imu[time_sync_col_renamed], errors='coerce')
            if df_imu[time_sync_col_renamed].isnull().any():
                logger.warning(f"NaNs introduced in {time_sync_col_renamed} for {file_path.name} after numeric conversion. Check source data.")
        else:
            logger.warning(f"Could not find or convert time_from_sync column for {imu_type} in {file_path.name}. Expected based on mapping: {time_sync_col_renamed}")

        # Convert actual data columns (e.g., ax, ay, az) to numeric
        data_axes_cols = SENSOR_INFO.get(imu_type, {}).get('axes', [])
        for col_name in data_axes_cols:
            if col_name in df_imu.columns:
                df_imu[col_name] = pd.to_numeric(df_imu[col_name], errors='coerce')
                if df_imu[col_name].isnull().any():
                    logger.warning(f"NaNs introduced in data column {col_name} for {file_path.name} after numeric conversion. Check source data.")
            else:
                logger.warning(f"Expected data column {col_name} not found in {file_path.name} for IMU type {imu_type}.")

        t_col_name = column_map.get('t_raw')
        if t_col_name in df_imu.columns:
            try:
                # df_imu[t_col_name + '_dt'] = pd.to_datetime(df_imu[t_col_name], unit='s', errors='coerce')
                # # If all NaT, maybe it's a different format or already datetime-like string
                # if df_imu[t_col_name + '_dt'].isnull().all():
                #     df_imu[t_col_name + '_dt'] = pd.to_datetime(df_imu[t_col_name], errors='coerce')
                pass # Keep timestamps as floats
            except Exception as e:
                logger.warning(f"Could not parse timestamp for {file_path.name}, column {t_col_name}: {e}")
        
        return df_imu

    def get_sensor_info(self, imu_type: str) -> Optional[Dict[str, Any]]:
        """Returns SENSOR_INFO for a given IMU type."""
        return SENSOR_INFO.get(imu_type)

    def validate_data(self) -> List[str]:
        """
        Validate the loaded experiment data.
        
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Validate GPS data
        if self.gps_data is None:
            issues.append("GPS data missing")
        elif self.gps_data.empty:
            issues.append("GPS data is empty")
        
        # Validate IMU data availability
        for sensor in self.available_sensors:
            for imu_type in self.available_imu_types[sensor]:
                if sensor not in self._imu_file_paths or imu_type not in self._imu_file_paths[sensor]:
                    issues.append(f"{sensor}/{imu_type} file path missing")
        
        # Check for minimum data requirements
        if not self.available_sensors:
            issues.append("No IMU sensors available")
        
        if self.time_of_day is None and any('morning' in s or 'afternoon' in s for s in self.available_imu_types.values()):
            issues.append("Time of day not determined but may be relevant")
        
        return issues

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of available data"""
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type,
            'time_of_day': self.time_of_day,
            'gps_data_available': self.gps_data is not None,
            'gps_rows': len(self.gps_data) if self.gps_data is not None else 0,
            'available_sensors': self.available_sensors,
            'imu_types_by_sensor': self.available_imu_types,
            'misc_files': list(self.misc_files.keys()),
            'file_structure_type': self.file_structure_type,
            'validation_issues': self.validate_data()
        }
        return summary

# --- Utility Functions ---

def _detect_file_structure(experiment_dir: Path) -> Tuple[Optional[str], Optional[Path]]:
    """
    Detect the file structure type for IMU data.
    
    Returns:
        Tuple of (structure_type, base_imu_path)
    """
    structures = {
        'standard': experiment_dir / "IMU",
        'flat': experiment_dir,
        'alternative': experiment_dir / "Sensors"
    }
    
    for name, path in structures.items():
        if path.exists():
            # Check if this path contains sensor directories
            sensor_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("Sensor_")]
            if sensor_dirs:
                logger.debug(f"Detected {name} structure with {len(sensor_dirs)} sensors at {path}")
                return name, path
    
    logger.warning(f"No recognizable IMU structure found in {experiment_dir}")
    return None, None

def _find_experiment_path(
    root_paths_to_experiments_base: List[str],
    experiment_type: str,
    experiment_name: str
) -> Optional[Tuple[Path, Path, Optional[str]]]:
    """
    Finds the path to a specific experiment directory.

    Returns:
        Tuple (base_path, experiment_dir_path, time_of_day) or None if not found.
        base_path is the .../02_Evaluation_Experiments part.
    """
    for root_candidate in root_paths_to_experiments_base:
        base = Path(root_candidate)
        if not base.is_dir():
            continue

        exp_type_path = base / experiment_type
        if not exp_type_path.is_dir():
            continue

        # Check direct path
        potential_path = exp_type_path / experiment_name
        if potential_path.is_dir():
            return base, potential_path, None

        # Check under morning/afternoon
        for tod_folder in ["morning", "afternoon"]:
            potential_path_tod = exp_type_path / tod_folder / experiment_name
            if potential_path_tod.is_dir():
                return base, potential_path_tod, tod_folder
    return None


def load_experiment_data(
    experiment_type: str,
    experiment_name: str,
    root_paths_to_experiments_base: List[str] = None
) -> Optional[ExperimentData]:
    """
    Loads all data for a given experiment into an ExperimentData object.

    Args:
        experiment_type: The type/category of the experiment (e.g., "1a_1_Minimum_Radius_Turn").
        experiment_name: The specific name of the experiment run (e.g., "007_Fast_stbd_turn_1").
        root_paths_to_experiments_base: List of base directories to search within.

    Returns:
        An ExperimentData object or None if the experiment path is not found.
    """
    if root_paths_to_experiments_base is None:
        root_paths_to_experiments_base = DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE
    
    found_path_info = _find_experiment_path(root_paths_to_experiments_base, experiment_type, experiment_name)
    if not found_path_info:
        logger.error(f"Experiment directory not found for {experiment_type}/{experiment_name}")
        return None

    base_path, experiment_dir, time_of_day = found_path_info
    
    exp_data = ExperimentData(experiment_name, experiment_type, base_path, experiment_dir, time_of_day)
    logger.info(f"Loading data for: {experiment_dir}")

    # Detect file structure
    structure_type, imu_base_path = _detect_file_structure(experiment_dir)
    exp_data.file_structure_type = structure_type

    # 1. Load GPS Data with enhanced error handling
    try:
        _load_gps_data(exp_data, experiment_dir, experiment_name)
    except Exception as e:
        logger.error(f"Error loading GPS data: {e}")

    # 2. Load IMU Data (with lazy loading setup)
    try:
        _setup_imu_data_loading(exp_data, experiment_dir, experiment_name, imu_base_path)
    except Exception as e:
        logger.error(f"Error setting up IMU data loading: {e}")

    # 3. Load miscellaneous files
    try:
        _load_misc_files(exp_data, experiment_dir)
    except Exception as e:
        logger.error(f"Error loading miscellaneous files: {e}")

    # Log summary
    summary = exp_data.get_summary()
    logger.info(f"Loaded experiment: {summary['experiment_name']}")
    logger.info(f"Available sensors: {summary['available_sensors']}")
    if summary['validation_issues']:
        logger.warning(f"Validation issues: {summary['validation_issues']}")

    return exp_data

def _load_gps_data(exp_data: ExperimentData, experiment_dir: Path, experiment_name: str):
    """Loads GPS data for the experiment."""
    gps_file_pattern = experiment_dir / "GPS" / f"GPS_{experiment_name}.csv"
    gps_files = list(experiment_dir.glob(f"GPS/GPS_{experiment_name}*.csv"))

    if not gps_files:
        logger.info(f"No GPS file found for {experiment_name} using pattern {gps_file_pattern}")
        exp_data.gps_data = None
        return

    if len(gps_files) > 1:
        logger.warning(f"Multiple GPS files found for {experiment_name}, using the first one: {gps_files[0]}")
    
    file_path = gps_files[0]
    try:
        df = pd.read_csv(file_path, na_filter=False) # Keep na_filter=False if specific strings mean NaN
        logger.info(f"Successfully read GPS data from {file_path}")

        # --- Robustly find and process 'time_from_sync' column ---
        actual_time_sync_col = None
        expected_time_sync_col_lowercase = 'time_from_sync'

        for col in df.columns:
            if col.lower() == expected_time_sync_col_lowercase:
                actual_time_sync_col = col
                break
        
        if actual_time_sync_col:
            # Convert to numeric
            df[actual_time_sync_col] = pd.to_numeric(df[actual_time_sync_col], errors='coerce')
            # Rename to canonical name if it's different
            if actual_time_sync_col != expected_time_sync_col_lowercase:
                df.rename(columns={actual_time_sync_col: expected_time_sync_col_lowercase}, inplace=True)
            logger.info(f"Processed '{expected_time_sync_col_lowercase}' column in GPS data.")
        else:
            logger.warning(f"'{expected_time_sync_col_lowercase}' column not found (case-insensitively) in {file_path}. GPS data might lack proper sync time.")
        # --- End robust processing ---

        # Timestamp parsing (optional, if needed beyond sync time)
        if 'Timestamp' in df.columns:
            try:
                # Attempt to parse as Unix timestamp first
                df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
                # If all NaT, try inferring (slower)
                if df['Timestamp_dt'].isnull().all():
                    logger.debug(f"Unix parsing of 'Timestamp' failed for {file_path.name}, trying to infer format.")
                    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                
                if df['Timestamp_dt'].isnull().all():
                     logger.warning(f"Could not parse 'Timestamp' into datetime for {file_path.name} after multiple attempts.")
                else:
                    logger.debug(f"Parsed 'Timestamp' for {file_path.name}")

            except Exception as e:
                logger.warning(f"Could not parse 'Timestamp' for {file_path.name}: {e}")
        else:
            logger.info(f"'Timestamp' column not found in {file_path.name}")

        exp_data.gps_data = df
        logger.debug(f"Loaded GPS data for {experiment_name} with columns: {df.columns.tolist()}")

    except FileNotFoundError:
        logger.warning(f"GPS file not found: {file_path}")
        exp_data.gps_data = None
    except pd.errors.EmptyDataError:
        logger.warning(f"GPS file is empty: {file_path}")
        exp_data.gps_data = None
    except Exception as e:
        logger.error(f"Error loading GPS data from {file_path}: {e}")
        exp_data.gps_data = None

def _setup_imu_data_loading(exp_data: ExperimentData, experiment_dir: Path, experiment_name: str, imu_base_path: Optional[Path]):
    """Setup IMU data loading with lazy loading (store file paths instead of loading data)"""
    potential_imu_base_folders = [imu_base_path] if imu_base_path else [experiment_dir, experiment_dir / "IMU"]
    
    sensors_with_data = []
    
    for sensor_name in SENSOR_NAMES:
        sensor_path_found = None

        for imu_base_folder_candidate in potential_imu_base_folders:
            if not imu_base_folder_candidate:
                continue
            current_sensor_path = imu_base_folder_candidate / sensor_name
            if current_sensor_path.is_dir():
                sensor_path_found = current_sensor_path
                break
        
        if not sensor_path_found:
            continue

        logger.debug(f"  Found sensor directory: {sensor_path_found}")
        sensor_has_data = False
        
        for imu_type in IMU_DATA_TYPES:
            imu_file_name = f"{imu_type}_{experiment_name}.csv"
            imu_file_path = sensor_path_found / imu_file_name

            if imu_file_path.is_file():
                # Store file path for lazy loading instead of loading now
                exp_data._imu_file_paths[sensor_name][imu_type] = imu_file_path
                exp_data._loaded_imu[sensor_name][imu_type] = False  # Mark as not loaded yet
                exp_data.available_imu_types[sensor_name].append(imu_type)
                sensor_has_data = True
                logger.debug(f"    Found {imu_type} file: {imu_file_name}")
        
        if sensor_has_data:
            exp_data.available_sensors.append(sensor_name)
            sensors_with_data.append(sensor_name)
    
    if sensors_with_data:
        logger.info(f"  Setup lazy loading for sensors: {sensors_with_data}")
        # Log which IMU types are available for each sensor
        for sensor in sensors_with_data:
            types = exp_data.available_imu_types[sensor]
            logger.info(f"    {sensor}: {types}")

def _load_misc_files(exp_data: ExperimentData, experiment_dir: Path):
    """Load miscellaneous non-CSV files"""
    for item in experiment_dir.iterdir():
        if item.is_file() and item.suffix not in ['.csv']:
            exp_data.misc_files[item.name] = item

def list_experiment_types(root_paths_to_experiments_base: List[str] = None) -> List[str]:
    """Lists all available experiment type directories."""
    if root_paths_to_experiments_base is None:
        root_paths_to_experiments_base = DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE
    
    exp_types = set()
    for root_candidate in root_paths_to_experiments_base:
        base = Path(root_candidate)
        if base.is_dir():
            for item in base.iterdir():
                if item.is_dir() and not item.name.startswith('.'): # Basic check
                    # Further check if item.name matches typical experiment type patterns like "1a_1_..."
                    if item.name[0].isdigit() and '_' in item.name:
                         exp_types.add(item.name)
    return sorted(list(exp_types))

def list_experiments(experiment_type: str, root_paths_to_experiments_base: List[str] = None) -> List[str]:
    """Lists all experiment runs within a given experiment type."""
    if root_paths_to_experiments_base is None:
        root_paths_to_experiments_base = DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE
    
    experiments = set()
    for root_candidate in root_paths_to_experiments_base:
        base = Path(root_candidate)
        if not base.is_dir():
            continue
        
        exp_type_path = base / experiment_type
        if not exp_type_path.is_dir():
            continue

        # Check directly under experiment_type
        for item in exp_type_path.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name not in ["morning", "afternoon"]:
                # Assuming experiment names start with a digit (e.g., "007_...")
                if item.name[0].isdigit() and '_' in item.name:
                    experiments.add(item.name)
        
        # Check under morning/afternoon
        for tod_folder in ["morning", "afternoon"]:
            tod_path = exp_type_path / tod_folder
            if tod_path.is_dir():
                for item in tod_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.') :
                        if item.name[0].isdigit() and '_' in item.name:
                            experiments.add(item.name)
                            
    return sorted(list(experiments))

# Enhanced batch loading with parallel processing capability
def load_multiple_experiments(
    experiment_configs: List[Dict[str, str]], 
    root_paths_to_experiments_base: List[str] = None,
    max_workers: int = 4
) -> Dict[str, ExperimentData]:
    """
    Load multiple experiments in parallel.
    
    Args:
        experiment_configs: List of dictionaries with 'experiment_type' and 'experiment_name' keys
        root_paths_to_experiments_base: List of base directories to search within
        max_workers: Maximum number of threads for parallel loading
    
    Returns:
        Dictionary mapping experiment names to ExperimentData objects
    """
    if root_paths_to_experiments_base is None:
        root_paths_to_experiments_base = DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all loading tasks
        futures = {
            executor.submit(
                load_experiment_data, 
                config['experiment_type'], 
                config['experiment_name'],
                root_paths_to_experiments_base
            ): config 
            for config in experiment_configs
        }
        
        # Collect results
        for future in futures:
            config = futures[future]
            exp_name = config['experiment_name']
            try:
                result = future.result()
                if result is not None:
                    results[exp_name] = result
                    logger.info(f"Successfully loaded experiment: {exp_name}")
                else:
                    logger.error(f"Failed to load experiment: {exp_name}")
            except Exception as e:
                logger.error(f"Exception loading experiment {exp_name}: {e}")
    
    return results

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    print("Available experiment types:")
    exp_types = list_experiment_types()
    for etype in exp_types:
        print(f"- {etype}")

    if "1a_1_Minimum_Radius_Turn" in exp_types:
        print("\nExperiments in '1a_1_Minimum_Radius_Turn':")
        min_turn_exps = list_experiments("1a_1_Minimum_Radius_Turn")
        for exp_name in min_turn_exps:
            print(f"- {exp_name}")

        # Try loading a specific experiment from 1a_1
        if "007_Fast_stbd_turn_1" in min_turn_exps:
            print("\nLoading '007_Fast_stbd_turn_1' from '1a_1_Minimum_Radius_Turn':")
            
            # Create dummy structure for testing if run in an empty dir
            TEST_ROOT = Path("test_data_root/analysis-pipeline/02_Evaluation_Experiments")
            
            # Check if default paths exist, otherwise use a dummy for demonstration
            actual_roots_exist = any(Path(p).exists() for p in DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE)
            
            current_test_roots = DEFAULT_ROOT_PATHS_TO_EXPERIMENTS_BASE
            if not actual_roots_exist and Path.cwd().name != "01_analysis_pipeline":
                 print("\n!!! WARNING: Default data paths not found. Will try to create a dummy structure for demonstration. !!!")
                 print("!!! This example will be more meaningful if run from the root of your project, or if paths are adjusted. !!!")

                 TEST_ROOT.mkdir(parents=True, exist_ok=True)
                 exp_type_path = TEST_ROOT / "1a_1_Minimum_Radius_Turn" / "afternoon" / "007_Fast_stbd_turn_1"
                 exp_type_path.mkdir(parents=True, exist_ok=True)
                 (exp_type_path / "GPS").mkdir(exist_ok=True)
                 (exp_type_path / "GPS" / "GPS_007_Fast_stbd_turn_1.csv").write_text("Timestamp,Lat,Lon\n1678886400,50.1,-1.0\n1678886401,50.2,-1.1")
                 
                 sensor3_path = exp_type_path / "IMU" / "Sensor_3"
                 sensor3_path.mkdir(parents=True, exist_ok=True)
                 (sensor3_path / "accel_007_Fast_stbd_turn_1.csv").write_text("1.0,0.1,0.2,9.8,0.001\n1.1,0.1,0.2,9.8,0.011")
                 (sensor3_path / "gyro_007_Fast_stbd_turn_1.csv").write_text("1.0,1,2,3,0.001\n1.1,1,2,3,0.011")

                 current_test_roots = ["test_data_root/analysis-pipeline/02_Evaluation_Experiments"]

            exp_data_007 = load_experiment_data(
                experiment_type="1a_1_Minimum_Radius_Turn",
                experiment_name="007_Fast_stbd_turn_1",
                root_paths_to_experiments_base=current_test_roots
            )

            if exp_data_007:
                print(f"\nLoaded: {exp_data_007}")
                print(f"Summary: {exp_data_007.get_summary()}")
                
                if exp_data_007.get_gps_data() is not None:
                    print("GPS Data Sample:")
                    print(exp_data_007.get_gps_data().head(2))
                
                if "Sensor_3" in exp_data_007.available_sensors:
                    print("\nSensor_3 IMU Data (lazy loading demonstration):")
                    for imu_type in exp_data_007.available_imu_types["Sensor_3"]:
                        # This will trigger lazy loading
                        imu_df = exp_data_007.get_imu_data("Sensor_3", imu_type)
                        if imu_df is not None:
                            print(f"  {imu_type} data sample (cols: {list(imu_df.columns)}):")
                            print(imu_df.head(2))
                            
                            # Example accessing SENSOR_INFO for plotting guidance
                            info = exp_data_007.get_sensor_info(imu_type)
                            if info:
                                print(f"    Plot Y-label: {info['ylabel']}, Axes: {info['axes']}")
                
                print("\nMiscellaneous files found:")
                for fname, fpath in exp_data_007.misc_files.items():
                    print(f"  - {fname} (at {fpath})")

            else:
                print("Failed to load '007_Fast_stbd_turn_1'. Check paths and structure.")
        else:
            print("Experiment '007_Fast_stbd_turn_1' not listed in '1a_1_Minimum_Radius_Turn'. Skipping load example.")
            
    # Clean up dummy structure if created
    if Path("test_data_root").exists() and not actual_roots_exist:
        import shutil
        shutil.rmtree("test_data_root")
        print("\nCleaned up dummy test_data_root directory.")