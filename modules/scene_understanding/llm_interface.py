"""
LLM Interface
Interacts with large language models
"""

import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
from ..utils.logger import default_logger as logger


class LLMInterface:
    """LLM Interface class"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4-turbo",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize LLM interface
        
        Args:
            provider: Provider (openai, anthropic)
            model: Model name
            api_key: API key
            temperature: Temperature parameter
            max_tokens: Maximum tokens
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if provider == "openai":
            if api_key:
                self.client = OpenAI(api_key=api_key)
                logger.info(f"OpenAI client initialized, model: {model}")
            else:
                logger.warning("API key not provided, LLM functionality will be unavailable")
                self.client = None
        else:
            logger.warning(f"Unsupported LLM provider: {provider}")
            self.client = None
    
    def generate_scene_description(
        self,
        scene_graph_dict: Dict
    ) -> str:
        """
        Generate scene description
        
        Args:
            scene_graph_dict: Scene graph dictionary
            
        Returns:
            Scene description text
        """
        if self.client is None:
            return self._generate_fallback_description(scene_graph_dict)
        
        # Build prompt
        system_prompt = """你是一个3D场景理解助手。你会收到一个场景的结构化数据，包括物体和它们的空间关系。
请生成一个自然、详细的场景描述，包括：
1. 场景中有哪些主要物体
2. 这些物体的空间布局
3. 物体之间的关系
4. 场景的整体特征

用中文回答，语言要自然流畅。"""
        
        user_prompt = f"""场景数据：
{json.dumps(scene_graph_dict, ensure_ascii=False, indent=2)}

请描述这个场景。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            description = response.choices[0].message.content
            logger.info("Scene description generated successfully")
            return description
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_description(scene_graph_dict)
    
    def answer_query(
        self,
        scene_graph_dict: Dict,
        query: str
    ) -> Dict[str, Any]:
        """
        Answer queries about the scene
        
        Args:
            scene_graph_dict: Scene graph dictionary
            query: User query
            
        Returns:
            Answer dictionary
        """
        if self.client is None:
            return {
                'answer': 'Sorry, LLM functionality is not enabled.',
                'highlight_objects': [],
                'camera_suggestion': None
            }
        
        system_prompt = """你是一个3D场景查询助手。用户会询问关于场景的问题，你需要：
1. 基于场景数据回答问题
2. 指出需要高亮显示的物体ID（如果适用）
3. 建议最佳观察视角（如果适用）

以JSON格式回答，包含以下字段：
{
  "answer": "自然语言回答",
  "highlight_objects": [物体ID列表],
  "reasoning": "推理过程",
  "camera_suggestion": {
    "position": [x, y, z],
    "target": [x, y, z],
    "description": "视角描述"
  }
}"""
        
        user_prompt = f"""场景数据：
{json.dumps(scene_graph_dict, ensure_ascii=False, indent=2)}

用户问题：{query}

请回答。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Query answered successfully: {query}")
            return result
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return {
                'answer': f'Query processing error: {str(e)}',
                'highlight_objects': [],
                'camera_suggestion': None
            }
    
    def suggest_viewpoint(
        self,
        scene_graph_dict: Dict,
        focus_object_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Suggest observation viewpoint
        
        Args:
            scene_graph_dict: Scene graph dictionary
            focus_object_id: Focus object ID
            
        Returns:
            Viewpoint suggestion
        """
        # Simple rule-based method
        scene_bounds = scene_graph_dict['scene_bounds']
        center = scene_bounds['center']
        size = scene_bounds['size']
        
        # Default: overlook from above the scene center
        camera_distance = max(size) * 1.5
        
        suggestion = {
            'position': [
                center[0] + camera_distance * 0.5,
                center[1] + camera_distance * 0.7,
                center[2] + camera_distance * 0.5
            ],
            'target': center,
            'description': 'Overlook entire scene from above'
        }
        
        # If focus object specified, adjust viewpoint
        if focus_object_id is not None:
            for obj in scene_graph_dict['objects']:
                if obj['object_id'] == focus_object_id:
                    obj_pos = obj['position']
                    suggestion['target'] = obj_pos
                    suggestion['position'] = [
                        obj_pos[0] + 2.0,
                        obj_pos[1] + 1.5,
                        obj_pos[2] + 2.0
                    ]
                    suggestion['description'] = f'Focus on {obj["class_name"]} object'
                    break
        
        return suggestion
    
    def _generate_fallback_description(self, scene_graph_dict: Dict) -> str:
        """
        Generate fallback description (when LLM is unavailable)
        
        Args:
            scene_graph_dict: Scene graph dictionary
            
        Returns:
            Description text
        """
        objects = scene_graph_dict['objects']
        stats = scene_graph_dict['statistics']
        
        desc = f"This scene contains {stats['num_objects']} objects, belonging to {stats['num_classes']} classes.\n\n"
        
        # Count each class
        class_counts = {}
        for obj in objects:
            class_name = obj['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        desc += "Object distribution:\n"
        for class_name, count in class_counts.items():
            desc += f"- {count} {class_name}\n"
        
        desc += f"\nScene size is approximately {scene_graph_dict['scene_bounds']['size']} meters."
        
        return desc

