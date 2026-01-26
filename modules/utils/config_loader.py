"""
Configuration File Loader
Responsible for loading and parsing YAML configuration files
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Configuration loader class"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Configuration file path, uses default path if None
        """
        if config_path is None:
            # Default configuration file path
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate required configuration items
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        Validate configuration file completeness
        
        Args:
            config: Configuration dictionary
        """
        required_sections = ['paths', 'yolo', 'gaussian', 'projection']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Configuration file missing required section: {section}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration item, supports dot-separated nested keys
        
        Args:
            key_path: Configuration key path, e.g. "yolo.model_name"
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_paths(self) -> Dict[str, str]:
        """Get all path configurations"""
        return self.config.get('paths', {})
    
    def get_yolo_config(self) -> Dict[str, Any]:
        """Get YOLO configuration"""
        return self.config.get('yolo', {})
    
    def get_gaussian_config(self) -> Dict[str, Any]:
        """Get Gaussian configuration"""
        return self.config.get('gaussian', {})
    
    def get_projection_config(self) -> Dict[str, Any]:
        """Get projection configuration"""
        return self.config.get('projection', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get('llm', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.config.get('visualization', {})
    
    def update_config(self, key_path: str, value: Any):
        """
        Update configuration item
        
        Args:
            key_path: Configuration key path
            value: New value
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: str = None):
        """
        Save configuration to file
        
        Args:
            output_path: Output path, overwrites original file if None
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

