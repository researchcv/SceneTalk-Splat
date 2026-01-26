"""
3D Object Reconstructor
Reconstructs 3D objects from multi-view detections
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
import cv2

from ..object_detection.detection_result import Detection, DetectionResult
from ..utils.camera_utils import CameraUtils
from ..utils.logger import default_logger as logger


@dataclass
class Object3D:
    """3D Object class"""
    object_id: int                                    # Object ID
    class_name: str                                   # Class name
    class_id: int                                     # Class ID
    confidence: float                                 # Average confidence
    
    # 3D geometry properties
    position: np.ndarray                              # 3D center position [x, y, z]
    bbox_3d_min: np.ndarray                          # 3D bounding box minimum point
    bbox_3d_max: np.ndarray                          # 3D bounding box maximum point
    size: np.ndarray                                  # Size [w, h, d]
    
    # Associated 2D detections
    detections_2d: List[Detection] = field(default_factory=list)
    view_ids: List[int] = field(default_factory=list)  # Visible view IDs
    
    # Optional properties
    gaussian_indices: Optional[List[int]] = None      # Associated Gaussian point indices
    orientation: Optional[np.ndarray] = None          # Orientation (Euler angles)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'object_id': self.object_id,
            'class_name': self.class_name,
            'class_id': self.class_id,
            'confidence': float(self.confidence),
            'position': self.position.tolist(),
            'bbox_3d_min': self.bbox_3d_min.tolist(),
            'bbox_3d_max': self.bbox_3d_max.tolist(),
            'size': self.size.tolist(),
            'num_views': len(self.view_ids),
            'view_ids': self.view_ids,
        }
    
    def get_bbox_corners(self) -> np.ndarray:
        """
        Get 8 corner points of 3D bounding box
        
        Returns:
            Corner points array [8, 3]
        """
        min_pt = self.bbox_3d_min
        max_pt = self.bbox_3d_max
        
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],  # 0
            [max_pt[0], min_pt[1], min_pt[2]],  # 1
            [max_pt[0], max_pt[1], min_pt[2]],  # 2
            [min_pt[0], max_pt[1], min_pt[2]],  # 3
            [min_pt[0], min_pt[1], max_pt[2]],  # 4
            [max_pt[0], min_pt[1], max_pt[2]],  # 5
            [max_pt[0], max_pt[1], max_pt[2]],  # 6
            [min_pt[0], max_pt[1], max_pt[2]],  # 7
        ])
        
        return corners


class Object3DReconstructor:
    """3D Object Reconstructor class"""
    
    def __init__(
        self,
        renderer,
        min_views: int = 2,
        clustering_eps: float = 1.0,
        clustering_min_samples: int = 3
    ):
        """
        Initialize reconstructor
        
        Args:
            renderer: Gaussian renderer
            min_views: Minimum required number of views
            clustering_eps: DBSCAN clustering distance threshold
            clustering_min_samples: DBSCAN minimum samples
        """
        self.renderer = renderer
        self.min_views = min_views
        self.clustering_eps = clustering_eps
        self.clustering_min_samples = clustering_min_samples
    
    def reconstruct_objects_3d(
        self,
        detection_results: List[DetectionResult],
        cameras: List
    ) -> List[Object3D]:
        """
        Reconstruct 3D objects from multi-view detections
        
        Args:
            detection_results: Detection results from multiple views
            cameras: Corresponding camera list
            
        Returns:
            List of 3D objects
        """
        logger.info(f"Starting 3D object reconstruction from {len(detection_results)} views...")
        
        # 1. Collect 3D position estimates for all detections
        all_3d_detections = []
        
        for idx, (det_result, camera) in enumerate(zip(detection_results, cameras)):
            # Render depth map
            depth_map = self.renderer.render_depth_map(camera)
            
            for det in det_result.detections:
                # Estimate 3D position
                position_3d = self._estimate_3d_position(det, depth_map, camera)
                
                if position_3d is not None:
                    all_3d_detections.append({
                        'detection': det,
                        'position_3d': position_3d,
                        'view_id': idx,
                        'camera': camera
                    })
        
        logger.info(f"Collected {len(all_3d_detections)} 3D detections")
        
        # 2. Group by class
        detections_by_class = {}
        for det_3d in all_3d_detections:
            class_name = det_3d['detection'].class_name
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            detections_by_class[class_name].append(det_3d)
        
        # 3. Perform clustering matching for each class
        objects_3d = []
        object_id = 0
        
        for class_name, dets in detections_by_class.items():
            class_objects = self._cluster_and_merge_detections(dets, class_name)
            
            for obj in class_objects:
                obj.object_id = object_id
                objects_3d.append(obj)
                object_id += 1
        
        logger.info(f"Reconstruction complete, total {len(objects_3d)} 3D objects")
        
        return objects_3d
    
    def _estimate_3d_position(
        self,
        detection: Detection,
        depth_map: np.ndarray,
        camera
    ) -> Optional[np.ndarray]:
        """
        Estimate 3D position from 2D detection and depth map
        
        Args:
            detection: 2D detection
            depth_map: Depth map
            camera: Camera
            
        Returns:
            3D position [x, y, z] or None
        """
        if depth_map is None:
            return None
        
        # Use detection box center
        center_x, center_y = detection.center
        x, y = int(center_x), int(center_y)
        
        # Check boundaries
        if not (0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]):
            return None
        
        # Get center depth
        depth = depth_map[y, x]
        
        # If center depth invalid, try neighborhood average
        if depth <= 0 or np.isnan(depth) or np.isinf(depth):
            x1, y1, x2, y2 = map(int, detection.bbox)
            region = depth_map[max(0, y1):min(depth_map.shape[0], y2),
                              max(0, x1):min(depth_map.shape[1], x2)]
            valid_depths = region[region > 0]
            if len(valid_depths) > 0:
                depth = np.median(valid_depths)
            else:
                return None
        
        # Back-project to 3D
        position_3d = CameraUtils.pixel_to_3d(center_x, center_y, depth, camera)
        
        return position_3d
    
    def _cluster_and_merge_detections(
        self,
        detections_3d: List[Dict],
        class_name: str
    ) -> List[Object3D]:
        """
        Cluster and merge multi-view detections of the same object
        
        Args:
            detections_3d: 3D detection list
            class_name: Class name
            
        Returns:
            List of merged 3D objects
        """
        if len(detections_3d) < self.min_views:
            return []
        
        # Extract 3D positions
        positions = np.array([d['position_3d'] for d in detections_3d])
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=self.clustering_min_samples
        ).fit(positions)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        logger.info(f"Class {class_name}: found {n_clusters} clusters")
        
        # Merge each cluster
        objects = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_detections = [d for i, d in enumerate(detections_3d) if cluster_mask[i]]
            
            if len(cluster_detections) >= self.min_views:
                obj = self._merge_cluster_to_object(cluster_detections, class_name)
                if obj is not None:
                    objects.append(obj)
        
        return objects
    
    def _merge_cluster_to_object(
        self,
        cluster_detections: List[Dict],
        class_name: str
    ) -> Optional[Object3D]:
        """
        Merge clustered detections into a single 3D object
        
        Args:
            cluster_detections: Detection list within cluster
            class_name: Class name
            
        Returns:
            3D object
        """
        if len(cluster_detections) == 0:
            return None
        
        # Calculate average position
        positions = np.array([d['position_3d'] for d in cluster_detections])
        avg_position = positions.mean(axis=0)
        
        # Calculate 3D bounding box
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        
        # Expand bounding box (considering object size)
        # Use 2D box size statistics to estimate
        box_sizes_2d = []
        for d in cluster_detections:
            width, height = d['detection'].get_width_height()
            box_sizes_2d.append((width, height))
        
        avg_size_2d = np.mean(box_sizes_2d, axis=0)
        # Simple estimation: assume 3D size proportional to 2D size
        size_factor = 0.01  # Adjustment factor
        expansion = avg_size_2d * size_factor
        
        bbox_min -= np.array([expansion[0], expansion[1], expansion[0]])
        bbox_max += np.array([expansion[0], expansion[1], expansion[0]])
        
        size = bbox_max - bbox_min
        
        # Average confidence
        confidences = [d['detection'].confidence for d in cluster_detections]
        avg_confidence = np.mean(confidences)
        
        # Collect 2D detections and views
        detections_2d = [d['detection'] for d in cluster_detections]
        view_ids = [d['view_id'] for d in cluster_detections]
        
        # Get class ID
        class_id = cluster_detections[0]['detection'].class_id
        
        obj = Object3D(
            object_id=-1,  # Assigned later
            class_name=class_name,
            class_id=class_id,
            confidence=float(avg_confidence),
            position=avg_position,
            bbox_3d_min=bbox_min,
            bbox_3d_max=bbox_max,
            size=size,
            detections_2d=detections_2d,
            view_ids=view_ids
        )
        
        return obj

