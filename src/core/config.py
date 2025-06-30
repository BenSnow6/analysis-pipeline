"""
Unified configuration management for the hovercraft analysis pipeline.

This module provides centralized configuration management with:
- Master configuration file (pipeline.yaml)
- Environment variable substitution
- Path validation
- Legacy compatibility
"""

import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration-related errors."""

    pass


class ConfigManager:
    """Enhanced configuration manager with master config support."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to master config file. If None, searches standard locations.
        """
        self._master_config_path = self._find_master_config(config_path)
        self._master_config = None
        self._sub_configs = {}
        self._environment = os.environ.get("HOVERCRAFT_ENV", "development")

    def _find_master_config(self, config_path: Optional[Path] = None) -> Path:
        """Find the master configuration file."""
        if config_path and config_path.exists():
            return config_path

        # Search standard locations
        search_paths = [
            Path.cwd() / "config" / "pipeline.yaml",
            Path.cwd().parent / "config" / "pipeline.yaml",
            Path(__file__).parent.parent.parent.parent / "config" / "pipeline.yaml",
            Path.home() / ".hovercraft" / "pipeline.yaml",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found master config at: {path}")
                return path

        # Fallback to legacy mode
        logger.warning("No master config found, using legacy configuration")
        return None

    @property
    def master_config(self) -> Dict[str, Any]:
        """Get the master configuration."""
        if self._master_config is None:
            if self._master_config_path:
                self._master_config = self._load_master_config()
            else:
                # Legacy mode - use old config structure
                self._master_config = self._create_legacy_config()
        return self._master_config

    def _load_master_config(self) -> Dict[str, Any]:
        """Load and process the master configuration file."""
        try:
            # Set PROJECT_ROOT environment variable if not set
            if "PROJECT_ROOT" not in os.environ:
                os.environ["PROJECT_ROOT"] = str(self._master_config_path.parent.parent)

            with open(self._master_config_path, "r") as f:
                config = yaml.safe_load(f)

            # Process environment variables
            config = self._substitute_env_vars(config)

            # Apply environment-specific overrides
            if self._environment in config.get("environments", {}):
                config = self._apply_overrides(
                    config, config["environments"][self._environment]
                )

            # Validate if enabled
            if config.get("features", {}).get("validate_on_load", True):
                self._validate_config(config)

            return config

        except Exception as e:
            raise ConfigError(f"Failed to load master config: {e}")

    def _create_legacy_config(self) -> Dict[str, Any]:
        """Create a master config from legacy file locations."""
        from . import paths

        return {
            "project": {"name": "hovercraft-analysis-pipeline", "version": "1.0.0"},
            "paths": {
                "project_root": str(paths.PROJECT_ROOT),
                "data_root": str(paths.DATA_DIR),
                "processed_data": {"aligned": str(paths.ALIGNED_DATA_DIR)},
            },
            "configs": {
                "experiments": {
                    "mapping": str(paths.EXPERIMENT_MAPPING_FILE),
                    "manifest": str(paths.EXPERIMENT_MANIFEST_FILE),
                },
                "sensors": {"orientations": str(paths.SENSOR_ORIENTATIONS_FILE)},
                "processing": {"orientation": str(paths.ORIENTATION_CONFIG_FILE)},
            },
            "features": {"use_legacy_paths": True},
        }

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in config.

        Supports ${VAR} and ${VAR:-default} syntax.
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Handle nested references first (e.g., ${processed_data.root})
            obj = self._resolve_references(obj)

            # Then handle environment variables
            pattern = re.compile(r"\$\{([^}]+)\}")

            def replacer(match):
                var_expr = match.group(1)

                # Handle default values (VAR:-default)
                if ":-" in var_expr:
                    var_name, default = var_expr.split(":-", 1)
                    return os.environ.get(var_name, default)
                else:
                    return os.environ.get(var_expr, match.group(0))

            return pattern.sub(replacer, obj)
        else:
            return obj

    def _resolve_references(self, value: str) -> str:
        """Resolve internal config references like ${paths.data_root}."""
        if not isinstance(value, str) or "${" not in value:
            return value

        pattern = re.compile(r"\$\{([^}:]+)\}")

        def replacer(match):
            ref_path = match.group(1)

            # Skip environment variables (those without dots)
            if "." not in ref_path:
                return match.group(0)

            # Navigate the config tree
            parts = ref_path.split(".")
            current = self._master_config

            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return match.group(0)  # Keep original if not found

            return str(current) if current is not None else match.group(0)

        # Resolve references iteratively (max 10 iterations to prevent infinite loops)
        for _ in range(10):
            new_value = pattern.sub(replacer, value)
            if new_value == value:
                break
            value = new_value

        return value

    def _apply_overrides(
        self, config: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply environment-specific overrides to config."""
        config = deepcopy(config)

        for key, value in overrides.items():
            # Handle dotted keys (e.g., "logging.level")
            parts = key.split(".")
            current = config

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return config

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and paths."""
        errors = []

        # Check required top-level keys
        required_keys = ["project", "paths", "configs"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Validate paths exist (for certain critical paths)
        if "paths" in config:
            critical_paths = ["data_root", "project_root"]
            for path_key in critical_paths:
                if path_key in config["paths"]:
                    path = Path(config["paths"][path_key])
                    if not path.exists() and not config.get("features", {}).get(
                        "auto_create_dirs", False
                    ):
                        errors.append(f"Path does not exist: {path_key} = {path}")

        if errors:
            raise ConfigError("Configuration validation failed:\n" + "\n".join(errors))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'paths.data_root')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split(".")
        current = self.master_config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def get_path(self, key: str) -> Path:
        """
        Get configuration path as Path object.

        Args:
            key: Path configuration key

        Returns:
            Path object
        """
        value = self.get(key)
        if value is None:
            raise ConfigError(f"Path not found in config: {key}")
        return Path(value)

    def load_sub_config(self, config_key: str) -> Dict[str, Any]:
        """
        Load a sub-configuration file referenced in master config.

        Args:
            config_key: Key in configs section (e.g., 'experiments.mapping')

        Returns:
            Loaded configuration dict
        """
        if config_key in self._sub_configs:
            return self._sub_configs[config_key]

        config_path = self.get(f"configs.{config_key}")
        if not config_path:
            raise ConfigError(f"Sub-config not found: {config_key}")

        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Sub-config file does not exist: {path}")

        if path.suffix == ".json":
            config = self._load_json(path)
        elif path.suffix in [".yaml", ".yml"]:
            config = self._load_yaml(path)
        else:
            raise ConfigError(f"Unsupported config format: {path.suffix}")

        self._sub_configs[config_key] = config
        return config

    def create_missing_dirs(self):
        """Create missing directories if auto_create_dirs is enabled."""
        if not self.get("features.auto_create_dirs", False):
            return

        # Create all directories mentioned in paths
        paths_config = self.get("paths", {})

        def create_paths(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    create_paths(value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(obj, str) and (
                prefix.endswith("_dir") or prefix.endswith("_root") or "data" in prefix
            ):
                path = Path(value)
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to create directory {path}: {e}")

        create_paths(paths_config)

    @staticmethod
    def _load_json(filepath: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {filepath}: {e}")
            return {}

    @staticmethod
    def _load_yaml(filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(filepath, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            return {}

    # Legacy compatibility properties
    @property
    def experiment_mapping(self) -> Dict[str, Any]:
        """Get experiment mapping configuration (legacy compatibility)."""
        try:
            return self.load_sub_config("experiments.mapping")
        except:
            # Fallback to old location
            from .paths import EXPERIMENT_MAPPING_FILE

            return self._load_json(EXPERIMENT_MAPPING_FILE)

    @property
    def sensor_orientations(self) -> Dict[str, Any]:
        """Get sensor orientations configuration (legacy compatibility)."""
        try:
            return self.load_sub_config("sensors.orientations")
        except:
            from .paths import SENSOR_ORIENTATIONS_FILE

            return self._load_json(SENSOR_ORIENTATIONS_FILE)

    @property
    def orientation_config(self) -> Dict[str, Any]:
        """Get orientation analysis configuration (legacy compatibility)."""
        try:
            return self.load_sub_config("processing.orientation")
        except:
            from .paths import ORIENTATION_CONFIG_FILE

            return self._load_yaml(ORIENTATION_CONFIG_FILE)

    @property
    def experiment_manifest(self) -> Dict[str, Any]:
        """Get experiment manifest (legacy compatibility)."""
        try:
            return self.load_sub_config("experiments.manifest")
        except:
            from .paths import EXPERIMENT_MANIFEST_FILE

            return self._load_yaml(EXPERIMENT_MANIFEST_FILE)

    def get_experiment_category(self, experiment_name: str) -> Optional[str]:
        """Get the evaluation category for an experiment."""
        for category, details in self.experiment_mapping.get(
            "evaluation_experiments", {}
        ).items():
            all_experiments = details.get("morning", []) + details.get("afternoon", [])
            if experiment_name in all_experiments:
                return category
        return None

    def get_sensor_orientation(self, sensor_name: str) -> Optional[Dict[str, Any]]:
        """Get orientation configuration for a specific sensor."""
        return self.sensor_orientations.get("sensors", {}).get(sensor_name)

    def reload(self):
        """Reload all configuration files."""
        self._master_config = None
        self._sub_configs = {}
        logger.info("Configuration reloaded")


# Keep the old Config class for backward compatibility
class Config(ConfigManager):
    """Legacy Config class for backward compatibility."""

    def __init__(self):
        """Initialize in legacy mode."""
        super().__init__(config_path=None)


# Global configuration instance
_config_manager = None


def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Legacy compatibility
config = Config()
