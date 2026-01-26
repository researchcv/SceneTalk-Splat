"""
File Manager
Responsible for creating and managing output directory structure
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class FileManager:
    """File Manager class"""
    
    def __init__(self, output_root: str, scene_name: str = None):
        """
        Initialize file manager
        
        Args:
            output_root: Output root directory
            scene_name: Scene name, uses timestamp if None
        """
        if scene_name is None:
            scene_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.output_root = Path(output_root)
        self.scene_name = scene_name
        self.scene_dir = self.output_root / scene_name
        
        # Create directory structure
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create standard directory structure"""
        # Main directories
        self.dirs = {
            'root': self.scene_dir,
            'original': self.scene_dir / '1_original_images',
            'yolo': self.scene_dir / '2_yolo_detection',
            'yolo_vis': self.scene_dir / '2_yolo_detection' / 'visualizations',
            'rendered': self.scene_dir / '3_gaussian_rendered',
            'rendered_train': self.scene_dir / '3_gaussian_rendered' / 'train_views',
            'rendered_test': self.scene_dir / '3_gaussian_rendered' / 'test_views',
            'rendered_novel': self.scene_dir / '3_gaussian_rendered' / 'novel_views',
            'projected': self.scene_dir / '4_projected_detection',
            'projected_comparison': self.scene_dir / '4_projected_detection' / 'comparison',
            'scene_understanding': self.scene_dir / '5_scene_understanding',
            'scene_query': self.scene_dir / '5_scene_understanding' / 'query_results',
            'report': self.scene_dir / '6_comprehensive_report',
            'report_figures': self.scene_dir / '6_comprehensive_report' / 'figures',
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_dir(self, dir_name: str) -> Path:
        """
        Get specified directory path
        
        Args:
            dir_name: Directory name (e.g. 'yolo', 'rendered')
            
        Returns:
            Directory path
        """
        if dir_name not in self.dirs:
            raise ValueError(f"Unknown directory name: {dir_name}")
        return self.dirs[dir_name]
    
    def get_path(self, dir_name: str, filename: str) -> Path:
        """
        Get complete file path
        
        Args:
            dir_name: Directory name
            filename: File name
            
        Returns:
            Complete file path
        """
        return self.get_dir(dir_name) / filename
    
    def save_json(self, data: Dict, dir_name: str, filename: str):
        """
        Save JSON file
        
        Args:
            data: Data to save
            dir_name: Directory name
            filename: File name
        """
        filepath = self.get_path(dir_name, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_json(self, dir_name: str, filename: str) -> Dict:
        """
        Load JSON file
        
        Args:
            dir_name: Directory name
            filename: File name
            
        Returns:
            Loaded data
        """
        filepath = self.get_path(dir_name, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_files(self, dir_name: str, pattern: str = "*") -> List[Path]:
        """
        List files in directory
        
        Args:
            dir_name: Directory name
            pattern: File pattern (e.g. "*.png")
            
        Returns:
            File path list
        """
        dir_path = self.get_dir(dir_name)
        return sorted(dir_path.glob(pattern))
    
    def copy_file(self, src: str, dir_name: str, filename: str = None):
        """
        Copy file to specified directory
        
        Args:
            src: Source file path
            dir_name: Target directory name
            filename: Target filename, uses source filename if None
        """
        src_path = Path(src)
        if filename is None:
            filename = src_path.name
        dst_path = self.get_path(dir_name, filename)
        shutil.copy2(src_path, dst_path)
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get file statistics for each directory
        
        Returns:
            Statistics dictionary
        """
        summary = {}
        for name, path in self.dirs.items():
            if path.exists():
                file_count = len(list(path.glob('*.*')))
                summary[name] = file_count
        return summary
    
    def create_readme(self, info: Dict[str, str]):
        """
        Create README file
        
        Args:
            info: Scene information dictionary
        """
        readme_path = self.scene_dir / "README.md"
        
        content = f"""# Scene Analysis Results: {self.scene_name}

## Basic Information
- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Scene Name: {self.scene_name}

## Directory Structure
- `1_original_images/`: Original input images
- `2_yolo_detection/`: YOLO object detection results
- `3_gaussian_rendered/`: Gaussian rendered images
- `4_projected_detection/`: Detection box projection results
- `5_scene_understanding/`: Scene understanding analysis
- `6_comprehensive_report/`: Comprehensive analysis report

## Statistics
"""
        
        # Add statistics
        summary = self.get_summary()
        for name, count in summary.items():
            content += f"- {name}: {count} files\n"
        
        # Add additional information
        if info:
            content += "\n## Details\n"
            for key, value in info.items():
                content += f"- {key}: {value}\n"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)

