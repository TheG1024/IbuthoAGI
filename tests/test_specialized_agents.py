"""
Tests for the specialized agents module.
"""
import pytest
from src.agents.specialized_agents import (
    BaseAgent,
    TechnicalArchitect,
    AgentCollaboration
)

def test_base_agent():
    agent = BaseAgent(
        name="Test Agent",
        expertise="Testing",
        responsibilities=["Unit testing", "Integration testing"]
    )
    
    assert agent.name == "Test Agent"
    assert agent.expertise == "Testing"
    assert len(agent.responsibilities) == 2
    assert len(agent.feedback_history) == 0

def test_technical_architect():
    architect = TechnicalArchitect()
    assert architect.name == "Technical Architect"
    assert "Architecture design" in architect.responsibilities
    assert "System integration" in architect.responsibilities

def test_agent_collaboration():
    collaboration = AgentCollaboration()
    
    # Test if all required agents are initialized
    required_agents = {
        "technical", "creative", "data", "domain",
        "qa", "risk", "user", "ethics"
    }
    assert set(collaboration.agents.keys()) == required_agents
    
    # Test collaborative feedback
    test_input = "Test problem statement"
    feedback = collaboration.get_collaborative_feedback(
        test_input,
        context={"stage": "analysis"}
    )
    
    assert isinstance(feedback, dict)
    assert len(feedback) == len(required_agents)
