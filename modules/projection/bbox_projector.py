"""
Bounding Box Projector
Projects 2D detection boxes to different views
"""

import numpy as np
import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from ..utils.camera_utils import CameraUtils
from ..utils.logger import default_logger as logger


@dataclass
class ProjectionResult:
    """Projection result"""
    projected_bbox: Optional[Tuple[float, float, float, float]]  # Projected 2D box
    visibility_score: float  # Visibility score [0-1]
    num_visible_points: int  # Number of visible points
    total_points: int        # Total number of points
    center_2d: Optional[Tuple[float, float]]  # Projected center point


class BBoxProjector:
    """Bounding box projector class"""
    
    def __init__(
        self,
        renderer,
        depth_sample_step: int = 10,
        min_visible_points: int = 4,
        visibility_threshold: float = 0.3
    ):
        """
        Initialize projector
        
        Args:
            renderer: Gaussian renderer
            depth_sample_step: Depth sampling step
            min_visible_points: Minimum visible points
            visibility_threshold: Visibility threshold
        """
        self.renderer = renderer
        self.depth_sample_step = depth_sample_step
        self.min_visible_points = min_visible_points
        self.visibility_threshold = visibility_threshold
    
    def project_bbox(
        self,
        bbox_2d: Tuple[float, float, float, float],
        source_camera,
        target_camera,
        depth_map: Optional[np.ndarray] = None,
        use_gaussian_depth: bool = True
    ) -> ProjectionResult:
        """
        Project 2D bounding box from source view to target view
        
        Args:
            bbox_2d: Source view 2D bounding box [x1, y1, x2, y2]
            source_camera: Source camera
            target_camera: Target camera
            depth_map: Optional depth map
            use_gaussian_depth: Whether to use Gaussian depth map
            
        Returns:
            Projection result
        """
        # 1. Get or render depth map
        if depth_map is None and use_gaussian_depth:
            depth_map = self.renderer.render_depth_map(source_camera)
        
        if depth_map is None:
            logger.warning("No depth map available, using default depth estimation")
            # Use scene average depth as estimation
            depth_map = np.ones((source_camera.image_height, 
                                source_camera.image_width)) * 5.0
        
        # 2. Sample 3D points within bounding box
        points_3d = self._sample_3d_points_from_bbox(
            bbox_2d, depth_map, source_camera
        )
        
        if len(points_3d) == 0:
            return ProjectionResult(
                projected_bbox=None,
                visibility_score=0.0,
                num_visible_points=0,
                total_points=0,
                center_2d=None
            )
        
        # 3. Project to target view
        points_2d = CameraUtils.project_3d_to_2d(points_3d, target_camera)
        
        # 4. Check visibility
        visible_mask = CameraUtils.check_point_in_view(
            points_2d, points_3d, target_camera
        )
        
        num_visible = visible_mask.sum()
        total_points = len(points_3d)
        visibility_score = num_visible / total_points if total_points > 0 else 0.0
        
        # 5. Calculate projected bounding box
        if num_visible >= self.min_visible_points:
            visible_points_2d = points_2d[visible_mask]
            
            x_min = visible_points_2d[:, 0].min()
            y_min = visible_points_2d[:, 1].min()
            x_max = visible_points_2d[:, 0].max()
            y_max = visible_points_2d[:, 1].max()
            
            # Add some margin
            margin = 5
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(target_camera.image_width, x_max + margin)
            y_max = min(target_camera.image_height, y_max + margin)
            
            projected_bbox = (float(x_min), float(y_min), 
                            float(x_max), float(y_max))
            
            # Calculate center point
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_2d = (float(center_x), float(center_y))
        else:
            projected_bbox = None
            center_2d = None
        
        return ProjectionResult(
            projected_bbox=projected_bbox,
            visibility_score=visibility_score,
            num_visible_points=int(num_visible),
            total_points=total_points,
            center_2d=center_2d
        )
    
    def _sample_3d_points_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        depth_map: np.ndarray,
        camera
    ) -> np.ndarray:
        """
        Sample 3D points from 2D bounding box and depth map
        
        Args:
            bbox: 2D bounding box [x1, y1, x2, y2]
            depth_map: Depth map [H, W]
            camera: Camera object
            
        Returns:
            3D point cloud [N, 3]
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        
        points_3d = []
        
        # Sample uniformly within bounding box
        for y in range(y1, y2, self.depth_sample_step):
            for x in range(x1, x2, self.depth_sample_step):
                if y < depth_map.shape[0] and x < depth_map.shape[1]:
                    depth = depth_map[y, x]
                    
                    # Filter invalid depth
                    if depth > 0 and not np.isnan(depth) and not np.isinf(depth):
                        # Back-project to 3D
                        point_3d = CameraUtils.pixel_to_3d(x, y, depth, camera)
                        points_3d.append(point_3d)
        
        # If too few sample points, add boundary sampling
        if len(points_3d) < self.min_visible_points:
            # Sample boundary points
            for x in [x1, x2]:
                for y in range(y1, y2, self.depth_sample_step):
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth = depth_map[y, x]
                        if depth > 0 and not np.isnan(depth):
                            point_3d = CameraUtils.pixel_to_3d(x, y, depth, camera)
                            points_3d.append(point_3d)
            
            for y in [y1, y2]:
                for x in range(x1, x2, self.depth_sample_step):
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth = depth_map[y, x]
                        if depth > 0 and not np.isnan(depth):
                            point_3d = CameraUtils.pixel_to_3d(x, y, depth, camera)
                            points_3d.append(point_3d)
        
        if len(points_3d) == 0:
            logger.warning(f"No valid 3D points found in bounding box {bbox}")
            return np.array([])
        
        return np.array(points_3d)
    
    def project_detections_to_view(
        self,
        detections: List,
        source_camera,
        target_camera,
        depth_map: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Project multiple detection results to target view
        
        Args:
            detections: Detection result list
            source_camera: Source camera
            target_camera: Target camera
            depth_map: Depth map
            
        Returns:
            Projection result list, each containing original detection and projection info
        """
        projected_results = []
        
        for det in detections:
            proj_result = self.project_bbox(
                det.bbox,
                source_camera,
                target_camera,
                depth_map
            )
            
            # Only keep projections with sufficient visibility
            if proj_result.visibility_score >= self.visibility_threshold:
                projected_results.append({
                    'detection': det,
                    'projection': proj_result,
                    'is_visible': True
                })
            else:
                projected_results.append({
                    'detection': det,
                    'projection': proj_result,
                    'is_visible': False
                })
        
        return projected_results
    
    def compute_projection_quality(
        self,
        projected_bbox: Tuple[float, float, float, float],
        ground_truth_bbox: Tuple[float, float, float, float]
    ) -> Dict[str, float]:
        """
        Calculate projection quality metrics
        
        Args:
            projected_bbox: Projected bounding box
            ground_truth_bbox: Ground truth bounding box
            
        Returns:
            Quality metrics dictionary
        """
        # Calculate IoU
        x1_min, y1_min, x1_max, y1_max = projected_bbox
        x2_min, y2_min, x2_max, y2_max = ground_truth_bbox
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * \
                    max(0, inter_y_max - inter_y_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        # Center distance
        center1 = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
        center2 = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
        center_dist = np.sqrt((center1[0] - center2[0])**2 + 
                             (center1[1] - center2[1])**2)
        
        # Size error
        size1 = (x1_max - x1_min) * (y1_max - y1_min)
        size2 = (x2_max - x2_min) * (y2_max - y2_min)
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0.0
        
        return {
            'iou': float(iou),
            'center_distance': float(center_dist),
            'size_ratio': float(size_ratio),
        }

