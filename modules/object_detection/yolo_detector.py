"""
YOLO Detector
Object detection using YOLOv8
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch

from .detection_result import Detection, DetectionResult
from ..utils.logger import default_logger as logger


class YOLODetector:
    """YOLO Detector class"""
    
    def __init__(
        self,
        model_name: str = 'yolov8x.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda',
        classes: Optional[List[int]] = None
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model name
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            device: Runtime device
            classes: List of class IDs to detect, None means all classes
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        
        # Get class names
        self.class_names = self.model.names
        
        # Generate color mapping
        self.colors = self._generate_colors(len(self.class_names))
        
        logger.info(f"YOLO model loaded successfully, supports {len(self.class_names)} classes")
    
    def _generate_colors(self, num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """
        Generate unique colors for each class
        
        Args:
            num_classes: Number of classes
            
        Returns:
            Color dictionary {class_id: (B, G, R)}
        """
        np.random.seed(42)  # Fixed random seed for consistent colors
        colors = {}
        for i in range(num_classes):
            colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors
    
    def detect(
        self,
        image_path: str,
        view_id: int = 0
    ) -> DetectionResult:
        """
        Detect objects in a single image
        
        Args:
            image_path: Image path
            view_id: View ID
            
        Returns:
            Detection result object
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return DetectionResult(
                image_name=Path(image_path).name,
                image_path=str(image_path),
                view_id=view_id,
                detections=[],
                image_size=(0, 0)
            )
        
        height, width = image.shape[:2]
        
        # Run YOLO detection
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )[0]
        
        # Parse detection results
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            cls_name = self.class_names[cls]
            
            # Create Detection object
            detection = Detection(
                class_name=cls_name,
                class_id=cls,
                confidence=conf,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(float((x1 + x2) / 2), float((y1 + y2) / 2))
            )
            detections.append(detection)
        
        logger.info(f"Detected {len(detections)} objects: {image_path}")
        
        return DetectionResult(
            image_name=Path(image_path).name,
            image_path=str(image_path),
            view_id=view_id,
            detections=detections,
            image_size=(width, height)
        )
    
    def detect_batch(
        self,
        image_paths: List[str],
        start_view_id: int = 0
    ) -> List[DetectionResult]:
        """
        Batch detect multiple images
        
        Args:
            image_paths: List of image paths
            start_view_id: Starting view ID
            
        Returns:
            List of detection results
        """
        results = []
        for idx, image_path in enumerate(image_paths):
            result = self.detect(image_path, view_id=start_view_id + idx)
            results.append(result)
        
        return results
    
    def visualize_detection(
        self,
        image_path: str,
        detection_result: DetectionResult,
        output_path: str = None,
        show_confidence: bool = True,
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Visualize detection results
        
        Args:
            image_path: Original image path
            detection_result: Detection result
            output_path: Output path, None means no save
            show_confidence: Whether to show confidence
            thickness: Bounding box line width
            font_scale: Font size
            
        Returns:
            Visualization image
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        vis_image = image.copy()
        
        # Draw each detection box
        for det in detection_result.detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = self.colors[det.class_id]
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            if show_confidence:
                label = f"{det.class_name} {det.confidence:.2f}"
            else:
                label = det.class_name
            
            # Calculate label background size
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )
            
            # Draw label background
            label_y1 = max(y1, label_h + 10)
            cv2.rectangle(
                vis_image,
                (x1, label_y1 - label_h - 10),
                (x1 + label_w, label_y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, label_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Add statistics info
        stats = detection_result.get_statistics()
        info_text = f"Detections: {stats['total_detections']}"
        cv2.putText(
            vis_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # Save image
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"Visualization result saved: {output_path}")
        
        return vis_image
    
    def get_summary_statistics(
        self,
        detection_results: List[DetectionResult]
    ) -> Dict:
        """
        Get summary statistics of all detection results
        
        Args:
            detection_results: List of detection results
            
        Returns:
            Statistics dictionary
        """
        total_detections = sum(len(res.detections) for res in detection_results)
        
        # Count each class
        class_counts = {}
        all_confidences = []
        
        for result in detection_results:
            for det in result.detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                all_confidences.append(det.confidence)
        
        return {
            'total_images': len(detection_results),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(detection_results) if detection_results else 0,
            'class_distribution': class_counts,
            'num_unique_classes': len(class_counts),
            'mean_confidence': float(np.mean(all_confidences)) if all_confidences else 0.0,
            'std_confidence': float(np.std(all_confidences)) if all_confidences else 0.0,
        }

