"""
Tests for the recommendation system.
"""
import pytest
import numpy as np
import torch
from datetime import datetime
from src.utils.recommendation_system import (
    RecommendationSystem,
    Recommendation,
    RecommendationGenerator
)

@pytest.fixture
def recommendation_system():
    """Create a recommendation system instance for testing."""
    return RecommendationSystem()

@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    return {
        'system_load': 0.85,
        'memory_usage': 0.75,
        'error_rate': 0.02,
        'api_latency': 150,
        'active_users': 1000
    }

@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    return [
        (
            {
                'system_load': 0.9,
                'memory_usage': 0.8,
                'error_rate': 0.03
            },
            ['optimize_performance', 'scale_resources']
        ),
        (
            {
                'api_latency': 200,
                'error_rate': 0.05,
                'active_users': 800
            },
            ['implement_caching', 'optimize_queries']
        )
    ]

def test_recommendation_generator_initialization():
    """Test initialization of recommendation generator."""
    generator = RecommendationGenerator(
        input_size=10,
        hidden_size=64,
        num_actions=5
    )
    
    assert isinstance(generator, torch.nn.Module)
    
    # Test forward pass
    dummy_input = torch.randn(1, 10)
    output = generator(dummy_input)
    assert output.shape == (1, 5)
    assert torch.all((output >= 0) & (output <= 1))

def test_embed_text(recommendation_system):
    """Test text embedding generation."""
    text = "optimize system performance"
    embedding = recommendation_system.embed_text(text)
    
    assert isinstance(embedding, np.ndarray)
    assert len(embedding.shape) == 1
    assert embedding.shape[0] > 0

def test_generate_recommendations(
    recommendation_system,
    sample_context,
    sample_training_data
):
    """Test recommendation generation."""
    # Initialize and train the system
    recommendation_system.initialize_recommendation_generator(
        input_size=recommendation_system.embed_text("test").shape[0],
        num_actions=len(sample_training_data[0][1])
    )
    
    # Add action embeddings
    for _, actions in sample_training_data:
        for action in actions:
            recommendation_system.action_embeddings[action] = \
                recommendation_system.embed_text(action)
    
    recommendation_system.train_recommendation_generator(sample_training_data)
    
    # Generate recommendations
    recommendations = recommendation_system.generate_recommendations(
        sample_context,
        top_k=2
    )
    
    assert len(recommendations) == 2
    assert all(isinstance(r, Recommendation) for r in recommendations)
    assert all(0 <= r.confidence <= 1 for r in recommendations)
    assert all(isinstance(r.explanation, str) for r in recommendations)
    assert all(isinstance(r.impact_areas, list) for r in recommendations)
    assert all(isinstance(r.prerequisites, list) for r in recommendations)
    assert all(
        r.priority in ['low', 'medium', 'high']
        for r in recommendations
    )

def test_context_history(recommendation_system, sample_context):
    """Test context history management."""
    actions = ['optimize_performance', 'scale_resources']
    
    # Update history
    recommendation_system.update_context_history(
        sample_context,
        actions
    )
    
    assert len(recommendation_system.context_history) == 1
    entry = recommendation_system.context_history[0]
    
    assert entry['context'] == sample_context
    assert entry['actions'] == actions
    assert isinstance(
        datetime.fromisoformat(entry['timestamp']),
        datetime
    )

def test_similar_contexts(recommendation_system, sample_context):
    """Test finding similar contexts."""
    # Add some history
    historical_contexts = [
        {
            'system_load': 0.82,
            'memory_usage': 0.78,
            'error_rate': 0.02
        },
        {
            'system_load': 0.45,
            'memory_usage': 0.30,
            'error_rate': 0.01
        }
    ]
    
    for context in historical_contexts:
        recommendation_system.update_context_history(
            context,
            ['some_action']
        )
    
    similar = recommendation_system.get_similar_contexts(
        sample_context,
        k=2
    )
    
    assert len(similar) == 2
    assert all(isinstance(c, dict) for c in similar)

def test_impact_area_determination(recommendation_system):
    """Test impact area determination."""
    action = "optimize_performance_scaling"
    context = {'system_load': 0.9}
    
    impact_areas = recommendation_system._determine_impact_areas(
        action,
        context
    )
    
    assert isinstance(impact_areas, list)
    assert 'performance' in impact_areas

def test_effort_and_priority_estimation(recommendation_system):
    """Test effort and priority estimation."""
    action = "implement_new_feature"
    impact_areas = ['functionality', 'performance']
    confidence = 0.85
    
    effort, priority = recommendation_system._estimate_effort_and_priority(
        action,
        impact_areas,
        confidence
    )
    
    assert effort in ['low', 'medium', 'high']
    assert priority in ['low', 'medium', 'high']
    assert effort == 'high'  # Based on 'implement' keyword
    assert priority == 'high'  # Based on impact areas and confidence
