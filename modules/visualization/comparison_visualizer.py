"""
Comparison Visualizer
Generates various comparison charts
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from ..utils.logger import default_logger as logger


class ComparisonVisualizer:
    """Comparison Visualizer class"""
    
    def __init__(self, dpi: int = 150):
        """
        Initialize visualizer
        
        Args:
            dpi: Image DPI
        """
        self.dpi = dpi
    
    def create_four_panel_comparison(
        self,
        original_img: np.ndarray,
        yolo_detected_img: np.ndarray,
        rendered_img: np.ndarray,
        projected_img: np.ndarray,
        output_path: str = None
    ) -> np.ndarray:
        """
        Create four-panel comparison
        
        Args:
            original_img: Original image
            yolo_detected_img: YOLO detected image
            rendered_img: Gaussian rendered image
            projected_img: Projected detection image
            output_path: Output path
            
        Returns:
            Concatenated image
        """
        # Ensure all images have same size
        h, w = original_img.shape[:2]
        
        # Resize other images
        yolo_detected_img = cv2.resize(yolo_detected_img, (w, h))
        rendered_img = cv2.resize(rendered_img, (w, h))
        projected_img = cv2.resize(projected_img, (w, h))
        
        # Add title
        def add_title(img, title, color=(0, 0, 0)):
            img_with_title = np.ones((h + 60, w, 3), dtype=np.uint8) * 255
            img_with_title[60:, :] = img
            
            # Add title text
            cv2.putText(
                img_with_title,
                title,
                (w//2 - len(title)*10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                2,
                cv2.LINE_AA
            )
            
            return img_with_title
        
        original_titled = add_title(original_img, "1. Original Image")
        yolo_titled = add_title(yolo_detected_img, "2. YOLO Detection")
        rendered_titled = add_title(rendered_img, "3. Gaussian Rendered")
        projected_titled = add_title(projected_img, "4. Projected Detection")
        
        # Concatenate: 2x2 layout
        top_row = np.hstack([original_titled, yolo_titled])
        bottom_row = np.hstack([rendered_titled, projected_titled])
        comparison = np.vstack([top_row, bottom_row])
        
        # Add separator lines
        h_total, w_total = comparison.shape[:2]
        
        # Vertical separator
        cv2.line(comparison, (w_total//2, 0), (w_total//2, h_total), (200, 200, 200), 3)
        
        # Horizontal separator
        cv2.line(comparison, (0, h_total//2), (w_total, h_total//2), (200, 200, 200), 3)
        
        # Save
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), comparison)
            logger.info(f"Comparison image saved: {output_path}")
        
        return comparison
    
    def create_side_by_side_comparison(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        title1: str = "Image 1",
        title2: str = "Image 2",
        output_path: str = None
    ) -> np.ndarray:
        """
        Create side-by-side comparison
        
        Args:
            img1: First image
            img2: Second image
            title1: Title 1
            title2: Title 2
            output_path: Output path
            
        Returns:
            Comparison image
        """
        # Ensure same size
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # Add title
        def add_title(img, title):
            img_with_title = np.ones((h + 50, w, 3), dtype=np.uint8) * 255
            img_with_title[50:, :] = img
            cv2.putText(img_with_title, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return img_with_title
        
        img1_titled = add_title(img1_resized, title1)
        img2_titled = add_title(img2_resized, title2)
        
        # Horizontal concatenation
        comparison = np.hstack([img1_titled, img2_titled])
        
        # Add separator lines
        h_total, w_total = comparison.shape[:2]
        cv2.line(comparison, (w_total//2, 0), (w_total//2, h_total), (200, 200, 200), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), comparison)
            logger.info(f"Comparison image saved: {output_path}")
        
        return comparison
    
    def create_detection_quality_plot(
        self,
        metrics: Dict[str, List[float]],
        output_path: str = None
    ):
        """
        Create detection quality statistics plot
        
        Args:
            metrics: Metrics dictionary, e.g. {'iou': [...], 'confidence': [...]}
            output_path: Output path
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.hist(values, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel(metric_name.upper())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric_name.upper()} Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
            ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Statistics plot saved: {output_path}")
        
        plt.close()
    
    def create_class_distribution_plot(
        self,
        class_counts: Dict[str, int],
        output_path: str = None
    ):
        """
        Create class distribution plot
        
        Args:
            class_counts: Class count dictionary
            output_path: Output path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Bar chart
        ax1.bar(classes, counts, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Object Class Distribution')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart
        ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Proportion')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Class distribution plot saved: {output_path}")
        
        plt.close()
    
    def create_projection_quality_heatmap(
        self,
        quality_matrix: np.ndarray,
        view_names: List[str],
        output_path: str = None
    ):
        """
        Create projection quality heatmap
        
        Args:
            quality_matrix: Quality matrix [N_views, N_views]
            view_names: View name list
            output_path: Output path
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(quality_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Set axes
        ax.set_xticks(np.arange(len(view_names)))
        ax.set_yticks(np.arange(len(view_names)))
        ax.set_xticklabels(view_names)
        ax.set_yticklabels(view_names)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values
        for i in range(len(view_names)):
            for j in range(len(view_names)):
                text = ax.text(j, i, f'{quality_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("Cross-View Projection Quality (IoU)")
        ax.set_xlabel("Target View")
        ax.set_ylabel("Source View")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('IoU Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Heatmap saved: {output_path}")
        
        plt.close()
    
    def draw_bbox_with_label(
        self,
        img: np.ndarray,
        bbox: Tuple[float, float, float, float],
        label: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw labeled bounding box on image
        
        Args:
            img: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            label: Label text
            color: Color (B, G, R)
            thickness: Line width
            
        Returns:
            Image with drawing
        """
        img_copy = img.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
        )
        
        cv2.rectangle(
            img_copy,
            (x1, y1 - label_h - 10),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        cv2.putText(
            img_copy,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness
        )
        
        return img_copy

