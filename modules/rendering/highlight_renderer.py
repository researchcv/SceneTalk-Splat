"""
Highlight Renderer
Supports highlighting specific objects
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from .gaussian_renderer import GaussianRendererWrapper
from ..utils.logger import default_logger as logger


class HighlightRenderer:
    """Highlight Renderer class"""
    
    def __init__(self, renderer: GaussianRendererWrapper):
        """
        Initialize highlight renderer
        
        Args:
            renderer: Gaussian renderer wrapper object
        """
        self.renderer = renderer
    
    def render_with_highlight(
        self,
        camera,
        highlight_mask: Optional[torch.Tensor] = None,
        highlight_color: Tuple[float, float, float] = (1.0, 1.0, 0.0),
        highlight_intensity: float = 0.5
    ) -> np.ndarray:
        """
        Render and highlight specified Gaussian points
        
        Args:
            camera: Camera object
            highlight_mask: Gaussian point mask [N], True means needs highlighting
            highlight_color: Highlight color (R, G, B), range [0, 1]
            highlight_intensity: Highlight intensity
            
        Returns:
            Highlighted image [H, W, 3]
        """
        # Render original image
        render_result = self.renderer.render_view(camera)
        rendered_image = render_result['render']  # [3, H, W]
        
        # Convert to numpy
        img_np = rendered_image.permute(1, 2, 0).cpu().numpy()
        
        # If highlight mask exists, perform highlighting
        if highlight_mask is not None:
            # TODO: Implement mask-based highlight rendering
            # This requires modifying the rendering process, temporarily using post-processing
            pass
        
        # Convert to uint8
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
        return img_np
    
    def render_with_bbox_overlay(
        self,
        camera,
        bboxes_2d: List[Tuple[float, float, float, float]],
        labels: List[str] = None,
        colors: List[Tuple[int, int, int]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Render image and overlay 2D bounding boxes
        
        Args:
            camera: Camera object
            bboxes_2d: 2D bounding box list [(x1, y1, x2, y2), ...]
            labels: Label list
            colors: Color list [(B, G, R), ...]
            thickness: Line width
            
        Returns:
            Annotated image [H, W, 3]
        """
        # Render base image
        render_result = self.renderer.render_view(camera)
        rendered_image = render_result['render']
        
        # Convert to numpy and BGR format
        img_np = rendered_image.permute(1, 2, 0).cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Default colors
        if colors is None:
            colors = [(0, 255, 0)] * len(bboxes_2d)  # Green
        
        # Draw each bounding box
        for idx, bbox in enumerate(bboxes_2d):
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[idx] if idx < len(colors) else (0, 255, 0)
            
            # Draw rectangle
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if labels and idx < len(labels):
                label = labels[idx]
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
                )
                
                # Label background
                cv2.rectangle(
                    img_bgr,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Label text
                cv2.putText(
                    img_bgr,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    thickness
                )
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb
    
    def render_with_3d_bbox(
        self,
        camera,
        bbox_3d_corners: List[np.ndarray],
        colors: List[Tuple[int, int, int]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Render image and overlay 3D bounding box projections
        
        Args:
            camera: Camera object
            bbox_3d_corners: 3D bounding box corner list, each is [8, 3]
            colors: Color list
            thickness: Line width
            
        Returns:
            Image with 3D boxes
        """
        from ..utils.camera_utils import CameraUtils
        
        # Render base image
        img_rgb = self.render_with_bbox_overlay(camera, [], [], [])
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Default colors
        if colors is None:
            colors = [(255, 0, 0)] * len(bbox_3d_corners)  # Red
        
        # 3D box edge connections
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]
        
        # Draw each 3D box
        for idx, corners_3d in enumerate(bbox_3d_corners):
            # Project to 2D
            corners_2d = CameraUtils.project_3d_to_2d(corners_3d, camera)
            
            # Check visibility
            visible = CameraUtils.check_point_in_view(corners_2d, corners_3d, camera)
            
            if visible.sum() < 4:  # At least 4 points visible
                continue
            
            color = colors[idx] if idx < len(colors) else (255, 0, 0)
            
            # Draw edges
            for edge in edges:
                if visible[edge[0]] and visible[edge[1]]:
                    pt1 = tuple(corners_2d[edge[0]].astype(int))
                    pt2 = tuple(corners_2d[edge[1]].astype(int))
                    cv2.line(img_bgr, pt1, pt2, color, thickness)
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_rgb

