"""
Object Detection Module
YOLO-based object detection and annotation
"""

from .yolo_detector import YOLODetector
from .detection_result import DetectionResult, Detection

__all__ = [
    'YOLODetector',
    'DetectionResult',
    'Detection',
]

