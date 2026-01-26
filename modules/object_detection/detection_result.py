"""
Detection Result Data Structures
Defines data classes for detection boxes and detection results
"""

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    """Single detection result"""
    
    # Basic information
    class_name: str          # Class name
    class_id: int            # Class ID
    confidence: float        # Confidence score
    
    # 2D bounding box [x1, y1, x2, y2]
    bbox: Tuple[float, float, float, float]
    
    # Center point [x, y]
    center: Tuple[float, float]
    
    # Optional 3D information (filled later)
    position_3d: Optional[Tuple[float, float, float]] = None  # 3D position
    size_3d: Optional[Tuple[float, float, float]] = None      # 3D size
    orientation: Optional[Tuple[float, float, float]] = None  # Orientation (Euler angles)
    
    # Associated Gaussian indices
    gaussian_indices: Optional[List[int]] = None
    
    # Visibility information
    visible_views: Optional[List[int]] = None  # List of visible view IDs
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def get_area(self) -> float:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_width_height(self) -> Tuple[float, float]:
        """Get bounding box width and height"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)
    
    def iou(self, other: 'Detection') -> float:
        """
        Calculate IoU with another detection box
        
        Args:
            other: Another detection object
            
        Returns:
            IoU value
        """
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = self.get_area()
        area2 = other.get_area()
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class DetectionResult:
    """Detection results for a single image"""
    
    image_name: str                    # Image name
    image_path: str                    # Image path
    view_id: int                       # View ID
    detections: List[Detection]        # Detection list
    image_size: Tuple[int, int]        # Image size (width, height)
    
    def __len__(self) -> int:
        """Return number of detections"""
        return len(self.detections)
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """
        Get all detections of a specified class
        
        Args:
            class_name: Class name
            
        Returns:
            Detection list
        """
        return [det for det in self.detections if det.class_name == class_name]
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """
        Filter detection results by confidence
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            New DetectionResult object
        """
        filtered_detections = [
            det for det in self.detections if det.confidence >= threshold
        ]
        
        return DetectionResult(
            image_name=self.image_name,
            image_path=self.image_path,
            view_id=self.view_id,
            detections=filtered_detections,
            image_size=self.image_size
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'image_name': self.image_name,
            'image_path': self.image_path,
            'view_id': self.view_id,
            'image_size': self.image_size,
            'num_detections': len(self.detections),
            'detections': [det.to_dict() for det in self.detections]
        }
    
    def get_statistics(self) -> dict:
        """
        Get detection statistics
        
        Returns:
            Statistics dictionary
        """
        class_counts = {}
        for det in self.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        return {
            'total_detections': len(self.detections),
            'class_distribution': class_counts,
            'mean_confidence': np.mean([det.confidence for det in self.detections]) if self.detections else 0.0,
            'min_confidence': min([det.confidence for det in self.detections]) if self.detections else 0.0,
            'max_confidence': max([det.confidence for det in self.detections]) if self.detections else 0.0,
        }

