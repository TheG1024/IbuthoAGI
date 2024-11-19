"""
Tests for the workflow and evaluation system.
"""
import pytest
from src.workflow.evaluation_system import WorkflowManager, EvaluationSystem

def test_workflow_initialization():
    workflow = WorkflowManager()
    
    # Test if all required stages are present
    required_stages = {
        "problem_analysis",
        "solution_ideation",
        "technical_assessment",
        "implementation_strategy"
    }
    assert set(workflow.stages.keys()) == required_stages
    
    # Test if stages have feedback loops
    for stage in workflow.stages.values():
        assert hasattr(stage, "feedback_loops")

def test_evaluation_system():
    evaluator = EvaluationSystem()
    
    # Test criteria weights
    total_weight = sum(
        criterion["weight"]
        for criterion in evaluator.criteria.values()
    )
    assert abs(total_weight - 1.0) < 0.0001  # Sum should be approximately 1
    
    # Test evaluation with sample data
    sample_data = {
        "analysis": {"completeness": 0.8},
        "solution": {"innovation": 0.7},
        "technical": {"feasibility": 0.9}
    }
    
    scores = evaluator.evaluate_solution(sample_data)
    assert isinstance(scores, dict)
    assert all(0 <= score <= 1 for score in scores.values())
