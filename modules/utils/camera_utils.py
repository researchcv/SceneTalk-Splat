"""
Camera Utility Functions
Provides camera-related calculations and conversion functions
"""

import numpy as np
import torch
from typing import Tuple, Optional


class CameraUtils:
    """Camera utility class"""
    
    @staticmethod
    def pixel_to_3d(x: float, y: float, depth: float, camera) -> np.ndarray:
        """
        Convert pixel coordinates and depth to 3D world coordinates
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            depth: Depth value
            camera: Camera object
            
        Returns:
            3D world coordinates [x, y, z]
        """
        # Get camera intrinsics
        fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
        cx = camera.image_width / 2
        cy = camera.image_height / 2
        
        # Pixel coordinates -> Camera coordinates
        x_cam = (x - cx) * depth / fx
        y_cam = (y - cy) * depth / fy
        z_cam = depth
        
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # Camera coordinates -> World coordinates
        if hasattr(camera, 'world_view_transform'):
            # Use inverse transform matrix
            w2c = camera.world_view_transform.cpu().numpy()
            c2w = np.linalg.inv(w2c.T)  # Note transpose
            point_world = c2w @ point_cam
            return point_world[:3]
        else:
            return point_cam[:3]
    
    @staticmethod
    def project_3d_to_2d(points_3d: np.ndarray, camera) -> np.ndarray:
        """
        Project 3D world coordinates to 2D pixel coordinates
        
        Args:
            points_3d: 3D point cloud [N, 3]
            camera: Camera object
            
        Returns:
            2D pixel coordinates [N, 2]
        """
        if len(points_3d.shape) == 1:
            points_3d = points_3d.reshape(1, -1)
        
        N = points_3d.shape[0]
        points_3d_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
        
        # World coordinates -> Camera coordinates
        if hasattr(camera, 'world_view_transform'):
            w2c = camera.world_view_transform.cpu().numpy().T
            points_cam = (w2c @ points_3d_homo.T).T
        else:
            points_cam = points_3d_homo
        
        # Camera coordinates -> Image coordinates
        fx = camera.image_width / (2 * np.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * np.tan(camera.FoVy / 2))
        cx = camera.image_width / 2
        cy = camera.image_height / 2
        
        # Avoid division by zero
        z = points_cam[:, 2]
        z[z == 0] = 1e-6
        
        x_img = fx * points_cam[:, 0] / z + cx
        y_img = fy * points_cam[:, 1] / z + cy
        
        points_2d = np.stack([x_img, y_img], axis=1)
        
        return points_2d
    
    @staticmethod
    def check_point_in_view(points_2d: np.ndarray, points_3d: np.ndarray, 
                           camera, depth_threshold: float = 0.01) -> np.ndarray:
        """
        Check if 3D points are within camera field of view
        
        Args:
            points_2d: 2D projection coordinates [N, 2]
            points_3d: 3D world coordinates [N, 3]
            camera: Camera object
            depth_threshold: Depth threshold
            
        Returns:
            Visibility mask [N]
        """
        # Check depth (must be in front of camera)
        if hasattr(camera, 'world_view_transform'):
            w2c = camera.world_view_transform.cpu().numpy().T
            N = points_3d.shape[0]
            points_3d_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
            points_cam = (w2c @ points_3d_homo.T).T
            depth_mask = points_cam[:, 2] > depth_threshold
        else:
            depth_mask = points_3d[:, 2] > depth_threshold
        
        # Check if within image bounds
        x_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera.image_width)
        y_mask = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera.image_height)
        
        visible_mask = depth_mask & x_mask & y_mask
        
        return visible_mask
    
    @staticmethod
    def compute_camera_distance(camera1, camera2) -> float:
        """
        Calculate distance between two cameras
        
        Args:
            camera1: First camera
            camera2: Second camera
            
        Returns:
            Distance value
        """
        pos1 = camera1.camera_center.cpu().numpy() if hasattr(camera1, 'camera_center') else np.zeros(3)
        pos2 = camera2.camera_center.cpu().numpy() if hasattr(camera2, 'camera_center') else np.zeros(3)
        
        return np.linalg.norm(pos1 - pos2)
    
    @staticmethod
    def get_camera_frustum_corners(camera, depth: float = 1.0) -> np.ndarray:
        """
        Get 8 corner points of camera frustum
        
        Args:
            camera: Camera object
            depth: Frustum depth
            
        Returns:
            Corner coordinates [8, 3]
        """
        # Pixel coordinates of four image corners
        corners_2d = np.array([
            [0, 0],
            [camera.image_width, 0],
            [camera.image_width, camera.image_height],
            [0, camera.image_height]
        ])
        
        # Near and far planes
        near_depth = camera.znear if hasattr(camera, 'znear') else 0.1
        far_depth = depth
        
        corners_3d = []
        
        for d in [near_depth, far_depth]:
            for corner in corners_2d:
                point_3d = CameraUtils.pixel_to_3d(corner[0], corner[1], d, camera)
                corners_3d.append(point_3d)
        
        return np.array(corners_3d)
    
    @staticmethod
    def interpolate_camera_path(camera_start, camera_end, num_steps: int):
        """
        Interpolate smooth path between two cameras
        
        Args:
            camera_start: Start camera
            camera_end: End camera
            num_steps: Interpolation steps
            
        Returns:
            Camera parameter list
        """
        # TODO: Implement camera path interpolation
        # This function can be used to generate smooth video transitions
        pass

