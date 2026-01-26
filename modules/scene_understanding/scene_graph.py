"""
Scene Graph
Builds and manages structured representation of scenes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

from ..projection.object_3d_reconstructor import Object3D
from ..utils.logger import default_logger as logger


@dataclass
class SpatialRelation:
    """Spatial relation"""
    subject_id: int          # Subject object ID
    predicate: str           # Relation type (on, near, left_of, etc.)
    object_id: int           # Object object ID
    distance: float          # Distance
    confidence: float = 1.0  # Confidence
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SceneGraph:
    """Scene Graph class"""
    
    def __init__(self, objects_3d: List[Object3D]):
        """
        Initialize scene graph
        
        Args:
            objects_3d: List of 3D objects
        """
        self.objects = {obj.object_id: obj for obj in objects_3d}
        self.relations: List[SpatialRelation] = []
        
        # Scene bounds
        self.scene_bounds = self._compute_scene_bounds()
        
        logger.info(f"Scene graph created, containing {len(self.objects)} objects")
    
    def _compute_scene_bounds(self) -> Dict[str, np.ndarray]:
        """Calculate scene bounds"""
        if len(self.objects) == 0:
            # Return default values (empty scene)
            return {
                'min': np.zeros(3),
                'max': np.zeros(3),
                'center': np.zeros(3),
                'size': np.zeros(3)
            }
        
        all_positions = np.array([obj.position for obj in self.objects.values()])
        
        return {
            'min': all_positions.min(axis=0),
            'max': all_positions.max(axis=0),
            'center': all_positions.mean(axis=0),
            'size': all_positions.max(axis=0) - all_positions.min(axis=0)
        }
    
    def add_relation(self, relation: SpatialRelation):
        """
        Add spatial relation
        
        Args:
            relation: Spatial relation object
        """
        self.relations.append(relation)
    
    def get_object_by_id(self, obj_id: int) -> Optional[Object3D]:
        """Get object by ID"""
        return self.objects.get(obj_id)
    
    def get_objects_by_class(self, class_name: str) -> List[Object3D]:
        """Get all objects of specified class"""
        return [obj for obj in self.objects.values() if obj.class_name == class_name]
    
    def get_relations_for_object(self, obj_id: int) -> List[SpatialRelation]:
        """Get all relations for specified object"""
        return [rel for rel in self.relations 
                if rel.subject_id == obj_id or rel.object_id == obj_id]
    
    def find_nearest_object(
        self,
        reference_obj_id: int,
        class_filter: Optional[str] = None
    ) -> Optional[Tuple[Object3D, float]]:
        """
        Find nearest object
        
        Args:
            reference_obj_id: Reference object ID
            class_filter: Class filter
            
        Returns:
            (Nearest object, distance) or None
        """
        ref_obj = self.objects.get(reference_obj_id)
        if ref_obj is None:
            return None
        
        min_dist = float('inf')
        nearest_obj = None
        
        for obj_id, obj in self.objects.items():
            if obj_id == reference_obj_id:
                continue
            
            if class_filter and obj.class_name != class_filter:
                continue
            
            dist = np.linalg.norm(ref_obj.position - obj.position)
            if dist < min_dist:
                min_dist = dist
                nearest_obj = obj
        
        if nearest_obj:
            return (nearest_obj, min_dist)
        return None
    
    def get_objects_in_radius(
        self,
        center: np.ndarray,
        radius: float,
        class_filter: Optional[str] = None
    ) -> List[Tuple[Object3D, float]]:
        """
        Get objects within radius
        
        Args:
            center: Center point [x, y, z]
            radius: Radius
            class_filter: Class filter
            
        Returns:
            [(object, distance), ...] list
        """
        results = []
        
        for obj in self.objects.values():
            if class_filter and obj.class_name != class_filter:
                continue
            
            dist = np.linalg.norm(obj.position - center)
            if dist <= radius:
                results.append((obj, dist))
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        
        return results
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            'objects': [obj.to_dict() for obj in self.objects.values()],
            'relations': [rel.to_dict() for rel in self.relations],
            'scene_bounds': {
                'min': self.scene_bounds['min'].tolist(),
                'max': self.scene_bounds['max'].tolist(),
                'center': self.scene_bounds['center'].tolist(),
                'size': self.scene_bounds['size'].tolist(),
            },
            'statistics': {
                'num_objects': len(self.objects),
                'num_relations': len(self.relations),
                'num_classes': len(set(obj.class_name for obj in self.objects.values())),
            }
        }
    
    def to_text_description(self) -> str:
        """Generate text description"""
        desc = f"Scene contains {len(self.objects)} objects:\n\n"
        
        # Group by class
        objects_by_class = {}
        for obj in self.objects.values():
            if obj.class_name not in objects_by_class:
                objects_by_class[obj.class_name] = []
            objects_by_class[obj.class_name].append(obj)
        
        for class_name, objs in objects_by_class.items():
            desc += f"- {len(objs)} {class_name}\n"
        
        desc += f"\nScene size: {self.scene_bounds['size']}\n"
        desc += f"Scene center: {self.scene_bounds['center']}\n"
        
        return desc

