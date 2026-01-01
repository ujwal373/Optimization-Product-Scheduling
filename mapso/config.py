"""
Configuration management for MAPSO

Handles loading and validation of configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for MAPSO"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        """Get path to default configuration file"""
        return str(Path(__file__).parent.parent / "configs" / "default.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            # Return minimal default config if file doesn't exist
            return self._get_minimal_config()

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override with environment variables
        config = self._apply_env_overrides(config)
        return config

    def _get_minimal_config(self) -> Dict[str, Any]:
        """Return minimal default configuration"""
        return {
            "logging": {
                "level": "INFO",
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            },
            "optimization": {
                "default_solver": "cpsat",
                "default_timeout": 300,
                "default_weights": {
                    "lateness": 0.4,
                    "setup_time": 0.2,
                    "cost": 0.2,
                    "energy": 0.2,
                },
            },
            "data": {
                "synthetic_path": "data/synthetic/",
                "output_path": "data/outputs/",
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": True,
            },
            "dashboard": {
                "port": 8501,
            },
        }

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # API configuration
        if os.getenv("API_HOST"):
            config.setdefault("api", {})["host"] = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config.setdefault("api", {})["port"] = int(os.getenv("API_PORT"))

        # Database (if needed in future)
        if os.getenv("DATABASE_URL"):
            config["database_url"] = os.getenv("DATABASE_URL")

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation)

        Args:
            key: Configuration key (e.g., "optimization.default_solver")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports nested keys with dot notation)

        Args:
            key: Configuration key (e.g., "optimization.default_solver")
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file

        Args:
            path: Path to save configuration. If None, uses original config_path
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        Config instance
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config


def reset_config() -> None:
    """Reset global configuration instance"""
    global _global_config
    _global_config = None
