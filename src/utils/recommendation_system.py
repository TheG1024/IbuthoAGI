"""
AI-powered recommendation system for IbuthoAGI.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

@dataclass
class Recommendation:
    """Recommendation with explanation and confidence."""
    action: str
    explanation: str
    confidence: float
    impact_areas: List[str]
    prerequisites: List[str]
    estimated_effort: str
    priority: str

class RecommendationGenerator(nn.Module):
    """Neural network for generating recommendations."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_actions: int = 10
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

class RecommendationSystem:
    """AI-powered recommendation system."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.recommendation_generator = None
        self.action_embeddings: Dict[str, np.ndarray] = {}
        self.context_history: List[Dict[str, Any]] = []
    
    def initialize_recommendation_generator(
        self,
        input_size: int,
        num_actions: int
    ) -> None:
        """Initialize the recommendation generator network."""
        self.recommendation_generator = RecommendationGenerator(
            input_size=input_size,
            num_actions=num_actions
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text using transformer model."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings[0]
    
    def train_recommendation_generator(
        self,
        training_data: List[Tuple[Dict[str, Any], List[str]]]
    ) -> None:
        """Train the recommendation generator on historical data."""
        X = []
        y = []
        
        for context, actions in training_data:
            # Create feature vector from context
            context_vector = self._create_context_vector(context)
            X.append(context_vector)
            
            # Create target vector (multi-label)
            target = np.zeros(len(self.action_embeddings))
            for action in actions:
                if action in self.action_embeddings:
                    target[list(self.action_embeddings.keys()).index(action)] = 1
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Train the model
        optimizer = torch.optim.Adam(
            self.recommendation_generator.parameters()
        )
        criterion = nn.BCELoss()
        
        self.recommendation_generator.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.recommendation_generator(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
    def _create_context_vector(
        self,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Create a feature vector from context information."""
        # Combine all context information into a single string
        context_str = ' '.join([
            f"{key}: {value}"
            for key, value in context.items()
        ])
        
        # Generate embedding
        return self.embed_text(context_str)
    
    def generate_recommendations(
        self,
        current_context: Dict[str, Any],
        top_k: int = 5
    ) -> List[Recommendation]:
        """Generate recommendations based on current context."""
        self.recommendation_generator.eval()
        
        # Create context vector
        context_vector = self._create_context_vector(current_context)
        context_tensor = torch.FloatTensor(context_vector).unsqueeze(0)
        
        # Generate action probabilities
        with torch.no_grad():
            action_probs = self.recommendation_generator(context_tensor)
        
        # Get top k recommendations
        top_k_probs, top_k_indices = torch.topk(action_probs[0], k)
        
        recommendations = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            action = list(self.action_embeddings.keys())[idx]
            
            # Generate explanation using context and action
            explanation = self._generate_explanation(
                action,
                current_context,
                float(prob)
            )
            
            # Determine impact areas
            impact_areas = self._determine_impact_areas(
                action,
                current_context
            )
            
            # Estimate effort and priority
            effort, priority = self._estimate_effort_and_priority(
                action,
                impact_areas,
                float(prob)
            )
            
            recommendations.append(Recommendation(
                action=action,
                explanation=explanation,
                confidence=float(prob),
                impact_areas=impact_areas,
                prerequisites=self._get_prerequisites(action),
                estimated_effort=effort,
                priority=priority
            ))
        
        return recommendations
    
    def _generate_explanation(
        self,
        action: str,
        context: Dict[str, Any],
        confidence: float
    ) -> str:
        """Generate a natural language explanation for the recommendation."""
        # Combine action and high-confidence context factors
        relevant_factors = [
            factor for factor, value in context.items()
            if self._is_relevant_to_action(action, factor, value)
        ]
        
        explanation = f"Recommended action '{action}' "
        
        if relevant_factors:
            explanation += f"based on {', '.join(relevant_factors)}. "
        
        explanation += f"This recommendation has {confidence:.1%} confidence."
        
        return explanation
    
    def _determine_impact_areas(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Determine areas that will be impacted by the recommendation."""
        impact_areas = set()
        
        # Map actions to potential impact areas
        impact_mapping = {
            'performance': ['optimize', 'tune', 'scale'],
            'reliability': ['monitor', 'backup', 'failover'],
            'security': ['encrypt', 'authenticate', 'authorize'],
            'functionality': ['implement', 'extend', 'integrate']
        }
        
        for area, keywords in impact_mapping.items():
            if any(keyword in action.lower() for keyword in keywords):
                impact_areas.add(area)
        
        return list(impact_areas)
    
    def _estimate_effort_and_priority(
        self,
        action: str,
        impact_areas: List[str],
        confidence: float
    ) -> Tuple[str, str]:
        """Estimate required effort and priority level."""
        # Effort estimation
        effort_keywords = {
            'high': ['implement', 'migrate', 'redesign'],
            'medium': ['optimize', 'integrate', 'extend'],
            'low': ['monitor', 'update', 'configure']
        }
        
        effort = 'medium'  # default
        for level, keywords in effort_keywords.items():
            if any(keyword in action.lower() for keyword in keywords):
                effort = level
                break
        
        # Priority calculation
        priority_score = len(impact_areas) * confidence
        if priority_score > 0.8:
            priority = 'high'
        elif priority_score > 0.5:
            priority = 'medium'
        else:
            priority = 'low'
        
        return effort, priority
    
    def _get_prerequisites(self, action: str) -> List[str]:
        """Determine prerequisites for an action."""
        # Map actions to prerequisites
        prerequisite_mapping = {
            'optimize': ['monitoring setup', 'performance baseline'],
            'scale': ['load testing', 'resource metrics'],
            'integrate': ['API documentation', 'authentication'],
            'implement': ['requirements document', 'design review']
        }
        
        prerequisites = []
        for keyword, reqs in prerequisite_mapping.items():
            if keyword in action.lower():
                prerequisites.extend(reqs)
        
        return list(set(prerequisites))
    
    def _is_relevant_to_action(
        self,
        action: str,
        factor: str,
        value: Any
    ) -> bool:
        """Determine if a context factor is relevant to an action."""
        # Convert action and factor to embeddings
        action_embedding = self.embed_text(action)
        factor_embedding = self.embed_text(f"{factor}: {value}")
        
        # Calculate similarity
        similarity = cosine_similarity(
            action_embedding.reshape(1, -1),
            factor_embedding.reshape(1, -1)
        )[0][0]
        
        return similarity > 0.5
    
    def update_context_history(
        self,
        context: Dict[str, Any],
        selected_actions: List[str]
    ) -> None:
        """Update context history with selected actions."""
        self.context_history.append({
            'context': context,
            'actions': selected_actions,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_similar_contexts(
        self,
        current_context: Dict[str, Any],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar historical contexts."""
        current_vector = self._create_context_vector(current_context)
        
        similarities = []
        for historical in self.context_history:
            historical_vector = self._create_context_vector(
                historical['context']
            )
            similarity = cosine_similarity(
                current_vector.reshape(1, -1),
                historical_vector.reshape(1, -1)
            )[0][0]
            similarities.append((similarity, historical))
        
        # Return top k similar contexts
        return [
            context for _, context in sorted(
                similarities,
                reverse=True
            )[:k]
        ]
