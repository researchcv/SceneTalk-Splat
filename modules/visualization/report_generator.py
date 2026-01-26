"""
Report Generator
Generates comprehensive analysis reports in HTML format
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from jinja2 import Template

from ..utils.logger import default_logger as logger


class ReportGenerator:
    """Report Generator class"""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_html_report(
        self,
        scene_name: str,
        summary: Dict[str, Any],
        detection_stats: Dict[str, Any],
        projection_metrics: Dict[str, Any],
        objects_3d: List[Dict],
        image_paths: Dict[str, List[str]] = None
    ) -> str:
        """
        Generate HTML report
        
        Args:
            scene_name: Scene name
            summary: Summary information
            detection_stats: Detection statistics
            projection_metrics: Projection metrics
            objects_3d: 3D object list
            image_paths: Image path dictionary
            
        Returns:
            Report file path
        """
        logger.info("Generating HTML report...")
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scene Analysis Report - {{ scene_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .timestamp {
            opacity: 0.9;
            font-size: 0.9em;
        }
        .section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card h3 {
            font-size: 2em;
            margin-bottom: 5px;
        }
        .stat-card p {
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #667eea;
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-item {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        .image-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .image-caption {
            padding: 10px;
            background: #f9f9f9;
            text-align: center;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric-label {
            font-weight: bold;
        }
        .metric-value {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 3D Scene Analysis Report</h1>
            <p class="timestamp">Scene: {{ scene_name }}</p>
            <p class="timestamp">Generated at: {{ timestamp }}</p>
        </header>
        
        <div class="section">
            <h2>📊 Overall Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{{ summary.total_images }}</h3>
                    <p>Total Images Processed</p>
                </div>
                <div class="stat-card">
                    <h3>{{ summary.total_detections }}</h3>
                    <p>Total Detections</p>
                </div>
                <div class="stat-card">
                    <h3>{{ summary.num_3d_objects }}</h3>
                    <p>3D Objects</p>
                </div>
                <div class="stat-card">
                    <h3>{{ summary.num_classes }}</h3>
                    <p>Object Classes</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Detection Statistics</h2>
            <div class="metric-row">
                <span class="metric-label">Average Detections per Image:</span>
                <span class="metric-value">{{ "%.2f"|format(detection_stats.avg_per_image) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Average Confidence:</span>
                <span class="metric-value">{{ "%.3f"|format(detection_stats.mean_confidence) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Confidence Standard Deviation:</span>
                <span class="metric-value">{{ "%.3f"|format(detection_stats.std_confidence) }}</span>
            </div>
            
            <h3 style="margin-top: 30px;">Class Distribution</h3>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {% for class_name, count in detection_stats.class_distribution.items() %}
                    <tr>
                        <td>{{ class_name }}</td>
                        <td>{{ count }}</td>
                        <td>{{ "%.1f"|format(count / summary.total_detections * 100) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>🎯 Projection Quality Evaluation</h2>
            <div class="metric-row">
                <span class="metric-label">Average IoU:</span>
                <span class="metric-value">{{ "%.3f"|format(projection_metrics.mean_iou) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Average Visibility Score:</span>
                <span class="metric-value">{{ "%.3f"|format(projection_metrics.mean_visibility) }}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Success Rate:</span>
                <span class="metric-value">{{ "%.1f"|format(projection_metrics.success_rate * 100) }}%</span>
            </div>
        </div>
        
        <div class="section">
            <h2>🏗️ 3D Object List</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Class</th>
                        <th>Confidence</th>
                        <th>Position (x, y, z)</th>
                        <th>Size (w, h, d)</th>
                        <th>Visible Views</th>
                    </tr>
                </thead>
                <tbody>
                    {% for obj in objects_3d %}
                    <tr>
                        <td>{{ obj.object_id }}</td>
                        <td>{{ obj.class_name }}</td>
                        <td>{{ "%.2f"|format(obj.confidence) }}</td>
                        <td>{{ "%.2f, %.2f, %.2f"|format(obj.position[0], obj.position[1], obj.position[2]) }}</td>
                        <td>{{ "%.2f, %.2f, %.2f"|format(obj.size[0], obj.size[1], obj.size[2]) }}</td>
                        <td>{{ obj.num_views }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        {% if image_paths %}
        <div class="section">
            <h2>📷 Visualization Results</h2>
            <div class="image-gallery">
                {% for img_path in image_paths.comparisons[:6] %}
                <div class="image-item">
                    <img src="{{ img_path }}" alt="Comparison Image">
                    <div class="image-caption">Comparison Image {{ loop.index }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <footer style="text-align: center; padding: 20px; color: #666;">
            <p>Generated by LLM-enhanced 3D Gaussian Scene Understanding System</p>
        </footer>
    </div>
</body>
</html>
"""
        
        # Render template
        template = Template(html_template)
        html_content = template.render(
            scene_name=scene_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
            detection_stats=detection_stats,
            projection_metrics=projection_metrics,
            objects_3d=objects_3d,
            image_paths=image_paths
        )
        
        # Save report
        report_path = self.output_dir / "report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_path}")
        
        return str(report_path)
    
    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Metrics dictionary
            filename: Filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metrics saved: {filepath}")

