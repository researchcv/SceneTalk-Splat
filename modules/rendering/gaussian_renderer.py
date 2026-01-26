"""
Gaussian Renderer Wrapper
Wraps original 3DGS rendering functionality
"""

import torch
import torchvision
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from argparse import Namespace
from ..utils.logger import default_logger as logger


class GaussianRendererWrapper:
    """Gaussian Renderer Wrapper class"""
    
    def __init__(
        self,
        model_path: str,
        source_path: str = None,
        sh_degree: int = 3,
        load_iteration: int = -1,
        white_background: bool = False
    ):
        """
        Initialize renderer
        
        Args:
            model_path: Trained model path
            source_path: Source dataset path
            sh_degree: Spherical harmonics degree
            load_iteration: Iteration to load
            white_background: Whether to use white background
        """
        self.model_path = Path(model_path)
        self.sh_degree = sh_degree
        self.white_background = white_background
        
        logger.info(f"Loading 3D Gaussian model: {model_path}")
        
        # If source_path not provided, try to infer from model path
        if source_path is None:
            cfg_file = self.model_path / "cfg_args"
            if cfg_file.exists():
                import re
                with open(cfg_file, 'r') as f:
                    cfg_content = f.read()
                    # Use regex to extract source_path
                    match = re.search(r"source_path='([^']*)'", cfg_content)
                    if match:
                        source_path = match.group(1)
                    else:
                        raise ValueError("Cannot infer source data path from config file, please specify source_path parameter manually")
            else:
                raise ValueError("Cannot find config file cfg_args, please specify source_path parameter manually")
        
        self.source_path = source_path
        logger.info(f"Using dataset path: {source_path}")
        
        # Create ModelParams style parameter object
        # Scene class needs an object with specific attributes, not a string
        dataset_args = Namespace(
            sh_degree=sh_degree,
            source_path=source_path,
            model_path=str(model_path),
            images="images",
            depths="",
            resolution=-1,
            white_background=white_background,
            train_test_exp=False,
            data_device="cuda",
            eval=False
        )
        
        # Load Gaussian model
        self.gaussians = GaussianModel(sh_degree)
        
        # Create scene (pass Namespace object instead of string)
        self.scene = Scene(dataset_args, self.gaussians, 
                         load_iteration=load_iteration, shuffle=False)
        
        # Set rendering parameters
        # PipelineParams needs parser argument, we create an object with default values directly
        self.pipeline = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False
        )
        
        # Set background color
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        logger.info(f"Model loaded, total {self.gaussians.get_xyz.shape[0]} Gaussian points")
    
    def render_view(
        self,
        camera,
        return_depth: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Render single view
        
        Args:
            camera: Camera object
            return_depth: Whether to return depth map
            
        Returns:
            Rendering result dictionary
        """
        with torch.no_grad():
            render_pkg = render(camera, self.gaussians, self.pipeline, self.background)
        
        result = {
            'render': render_pkg['render'],  # [3, H, W]
        }
        
        if return_depth:
            result['depth'] = render_pkg.get('depth', None)  # [1, H, W]
        
        return result
    
    def render_train_views(
        self,
        output_dir: str = None,
        return_images: bool = False,
        scale: float = 1.0
    ) -> Optional[List[np.ndarray]]:
        """
        Render all training views
        
        Args:
            output_dir: Output directory
            return_images: Whether to return image list
            scale: Resolution scale ratio
            
        Returns:
            If return_images is True, return image list
        """
        cameras = self.scene.getTrainCameras(scale)
        return self._render_camera_list(cameras, output_dir, "train", return_images)
    
    def render_test_views(
        self,
        output_dir: str = None,
        return_images: bool = False,
        scale: float = 1.0
    ) -> Optional[List[np.ndarray]]:
        """
        Render all test views
        
        Args:
            output_dir: Output directory
            return_images: Whether to return image list
            scale: Resolution scale ratio
            
        Returns:
            If return_images is True, return image list
        """
        cameras = self.scene.getTestCameras(scale)
        return self._render_camera_list(cameras, output_dir, "test", return_images)
    
    def _render_camera_list(
        self,
        cameras: List,
        output_dir: Optional[str],
        split_name: str,
        return_images: bool
    ) -> Optional[List[np.ndarray]]:
        """
        Render camera list
        
        Args:
            cameras: Camera list
            output_dir: Output directory
            split_name: Dataset split name
            return_images: Whether to return images
            
        Returns:
            Image list (if return_images is True)
        """
        images = [] if return_images else None
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Rendering {len(cameras)} {split_name} views...")
        
        for idx, camera in enumerate(tqdm(cameras, desc=f"Rendering {split_name} views")):
            # Render
            render_result = self.render_view(camera)
            rendered_image = render_result['render']  # [3, H, W]
            
            # Save or collect images
            if output_dir:
                filename = f"{idx:05d}_render.png"
                filepath = output_path / filename
                # Ensure image is in [0,1] range and save
                rendered_image_clamped = torch.clamp(rendered_image, 0.0, 1.0)
                torchvision.utils.save_image(rendered_image_clamped, filepath)
                logger.debug(f"Saved rendered image: {filepath}")
            
            if return_images:
                # Convert to numpy array
                img_np = rendered_image.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                images.append(img_np)
        
        logger.info(f"{split_name} views rendering complete")
        
        return images
    
    def get_train_cameras(self, scale: float = 1.0) -> List:
        """Get training camera list"""
        return self.scene.getTrainCameras(scale)
    
    def get_test_cameras(self, scale: float = 1.0) -> List:
        """Get test camera list"""
        return self.scene.getTestCameras(scale)
    
    def get_gaussian_positions(self) -> np.ndarray:
        """
        Get positions of all Gaussian points
        
        Returns:
            Position array [N, 3]
        """
        return self.gaussians.get_xyz.cpu().numpy()
    
    def get_gaussian_colors(self) -> np.ndarray:
        """
        Get colors of all Gaussian points
        
        Returns:
            Color array [N, 3]
        """
        features = self.gaussians.get_features
        # Simplified: only take DC component
        colors = features[:, :, 0].cpu().numpy()
        return colors
    
    def render_depth_map(self, camera) -> np.ndarray:
        """
        Render depth map
        
        Args:
            camera: Camera object
            
        Returns:
            Depth map [H, W]
        """
        with torch.no_grad():
            render_pkg = render(camera, self.gaussians, self.pipeline, self.background)
            depth = render_pkg.get('depth', None)
        
        if depth is not None:
            depth_np = depth.squeeze().cpu().numpy()
            return depth_np
        else:
            logger.warning("Depth map not available")
            return None
    
    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get scene bounds
        
        Returns:
            (min_bound, max_bound), each is [x, y, z]
        """
        positions = self.get_gaussian_positions()
        min_bound = positions.min(axis=0)
        max_bound = positions.max(axis=0)
        return min_bound, max_bound
    
    def estimate_scene_scale(self) -> float:
        """
        Estimate scene scale (maximum dimension)
        
        Returns:
            Scene scale
        """
        min_bound, max_bound = self.get_scene_bounds()
        scale = np.max(max_bound - min_bound)
        return float(scale)

