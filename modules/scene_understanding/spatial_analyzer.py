"""
Spatial Analyzer
Calculates spatial relations between objects
"""

import numpy as np
from typing import List, Dict, Tuple
from .scene_graph import SceneGraph, SpatialRelation, Object3D
from ..utils.logger import default_logger as logger


class SpatialAnalyzer:
    """Spatial Analyzer class"""
    
    def __init__(
        self,
        distance_threshold: float = 2.0,
        near_threshold: float = 1.0
    ):
        """
        Initialize spatial analyzer
        
        Args:
            distance_threshold: Spatial relation distance threshold
            near_threshold: Distance threshold for "near" relation
        """
        self.distance_threshold = distance_threshold
        self.near_threshold = near_threshold
    
    def analyze_scene(self, scene_graph: SceneGraph):
        """
        Analyze scene and add spatial relations
        
        Args:
            scene_graph: Scene graph object
        """
        logger.info("Starting spatial relation analysis...")
        
        objects = list(scene_graph.objects.values())
        
        # Calculate relations between all object pairs
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                relations = self._compute_pairwise_relations(obj1, obj2)
                for rel in relations:
                    scene_graph.add_relation(rel)
        
        logger.info(f"Spatial relation analysis complete, found {len(scene_graph.relations)} relations")
    
    def _compute_pairwise_relations(
        self,
        obj1: Object3D,
        obj2: Object3D
    ) -> List[SpatialRelation]:
        """
        Calculate spatial relations between two objects
        
        Args:
            obj1: Object 1
            obj2: Object 2
            
        Returns:
            Spatial relation list
        """
        relations = []
        
        # Calculate distance
        distance = np.linalg.norm(obj1.position - obj2.position)
        
        # 1. Distance relation (near/far)
        if distance < self.near_threshold:
            relations.append(SpatialRelation(
                subject_id=obj1.object_id,
                predicate='near',
                object_id=obj2.object_id,
                distance=distance
            ))
        
        # 2. Vertical relation (above/below/on)
        vertical_dist = obj1.position[1] - obj2.position[1]
        horizontal_dist = np.linalg.norm(obj1.position[[0,2]] - obj2.position[[0,2]])
        
        # Object 1 above object 2
        if vertical_dist > obj2.size[1] / 2:
            if horizontal_dist < (obj1.size[0] + obj2.size[0]) / 4:
                # Position aligned, possibly "on" relation
                if abs(vertical_dist - obj2.size[1]/2) < 0.3:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='on',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='above',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
        
        # Object 1 below object 2
        elif vertical_dist < -obj1.size[1] / 2:
            relations.append(SpatialRelation(
                subject_id=obj1.object_id,
                predicate='below',
                object_id=obj2.object_id,
                distance=distance
            ))
        
        # 3. Horizontal direction relation (left/right/front/back)
        if abs(vertical_dist) < max(obj1.size[1], obj2.size[1]) / 2:
            # On same horizontal plane
            dx = obj1.position[0] - obj2.position[0]
            dz = obj1.position[2] - obj2.position[2]
            
            if abs(dx) > abs(dz):
                # X-axis direction dominant
                if dx > 0:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='right_of',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='left_of',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
            else:
                # Z-axis direction dominant
                if dz > 0:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='behind',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
                else:
                    relations.append(SpatialRelation(
                        subject_id=obj1.object_id,
                        predicate='in_front_of',
                        object_id=obj2.object_id,
                        distance=distance
                    ))
        
        return relations
    
    def query_spatial_relation(
        self,
        scene_graph: SceneGraph,
        query: str
    ) -> List[Dict]:
        """
        Query spatial relations
        
        Args:
            scene_graph: Scene graph
            query: Query string, e.g. "chair on table"
            
        Returns:
            Query result list
        """
        # Simple query parsing
        query_lower = query.lower()
        
        results = []
        
        # Parse query type
        if ' on ' in query_lower:
            parts = query_lower.split(' on ')
            if len(parts) == 2:
                subject_class = parts[0].strip()
                object_class = parts[1].strip()
                
                # Find matching relations
                for rel in scene_graph.relations:
                    if rel.predicate == 'on':
                        subj_obj = scene_graph.get_object_by_id(rel.subject_id)
                        obj_obj = scene_graph.get_object_by_id(rel.object_id)
                        
                        if (subj_obj and obj_obj and
                            subject_class in subj_obj.class_name.lower() and
                            object_class in obj_obj.class_name.lower()):
                            results.append({
                                'subject': subj_obj.to_dict(),
                                'relation': rel.predicate,
                                'object': obj_obj.to_dict()
                            })
        
        elif ' near ' in query_lower:
            parts = query_lower.split(' near ')
            if len(parts) == 2:
                subject_class = parts[0].strip()
                object_class = parts[1].strip()
                
                for rel in scene_graph.relations:
                    if rel.predicate == 'near':
                        subj_obj = scene_graph.get_object_by_id(rel.subject_id)
                        obj_obj = scene_graph.get_object_by_id(rel.object_id)
                        
                        if (subj_obj and obj_obj and
                            subject_class in subj_obj.class_name.lower() and
                            object_class in obj_obj.class_name.lower()):
                            results.append({
                                'subject': subj_obj.to_dict(),
                                'relation': rel.predicate,
                                'object': obj_obj.to_dict(),
                                'distance': rel.distance
                            })
        
        return results
    
    def compute_object_statistics(
        self,
        scene_graph: SceneGraph
    ) -> Dict:
        """
        Calculate object statistics
        
        Args:
            scene_graph: Scene graph
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_objects': len(scene_graph.objects),
            'total_relations': len(scene_graph.relations),
            'classes': {},
            'relation_types': {},
            'avg_object_size': None,
            'scene_density': None,
        }
        
        # Class statistics
        for obj in scene_graph.objects.values():
            stats['classes'][obj.class_name] = stats['classes'].get(obj.class_name, 0) + 1
        
        # Relation type statistics
        for rel in scene_graph.relations:
            stats['relation_types'][rel.predicate] = stats['relation_types'].get(rel.predicate, 0) + 1
        
        # Average object size
        if len(scene_graph.objects) > 0:
            all_sizes = np.array([obj.size for obj in scene_graph.objects.values()])
            stats['avg_object_size'] = all_sizes.mean(axis=0).tolist()
        
        # Scene density (objects / scene volume)
        scene_volume = np.prod(scene_graph.scene_bounds['size'])
        if scene_volume > 0:
            stats['scene_density'] = len(scene_graph.objects) / scene_volume
        
        return stats

