"""
Tests for the core agent functionality and specialized agents.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
import json

from src.core.agent import Agent, AgentRole, AgentCapability
from src.agents.code_agent import CodeAgent
from src.agents.data_science_agent import DataScienceAgent
from src.agents.nlp_agent import NLPAgent
from src.utils.memory_manager import MemoryManager

@pytest.fixture
def memory_manager():
    return Mock(spec=MemoryManager)

@pytest.fixture
def mock_llm_chain():
    chain = Mock()
    chain.llm = Mock()
    chain.arun = asyncio.coroutine(lambda *args, **kwargs: json.dumps({
        "analysis": "Mock analysis result",
        "recommendations": ["Mock recommendation 1", "Mock recommendation 2"]
    }))
    return chain

@pytest.mark.asyncio
async def test_code_agent_analyze(memory_manager, mock_llm_chain):
    """Test CodeAgent's code analysis functionality."""
    agent = CodeAgent(
        agent_id="test_code_agent",
        memory_manager=memory_manager,
        llm_chain=mock_llm_chain
    )
    
    test_code = """
    def example_function():
        print("Hello, World!")
    """
    
    result = await agent.analyze_code(
        code=test_code,
        context={"purpose": "testing"},
        requirements={"style_check": True}
    )
    
    assert "session_id" in result
    assert "analysis" in result
    assert isinstance(result["analysis"], dict)

@pytest.mark.asyncio
async def test_data_science_agent_analysis(memory_manager, mock_llm_chain):
    """Test DataScienceAgent's data analysis functionality."""
    import pandas as pd
    import numpy as np
    
    agent = DataScienceAgent(
        agent_id="test_ds_agent",
        memory_manager=memory_manager,
        llm_chain=mock_llm_chain
    )
    
    # Create test dataset
    df = pd.DataFrame({
        'A': np.random.rand(10),
        'B': np.random.rand(10),
        'target': np.random.randint(0, 2, 10)
    })
    
    result = await agent.analyze_data(
        data=df,
        analysis_requirements={"correlation_analysis": True}
    )
    
    assert "session_id" in result
    assert "analysis" in result
    assert "basic_stats" in result["analysis"]
    assert "correlations" in result["analysis"]

@pytest.mark.asyncio
async def test_nlp_agent_text_analysis(memory_manager, mock_llm_chain):
    """Test NLPAgent's text analysis functionality."""
    with patch('spacy.load') as mock_spacy_load:
        mock_nlp = Mock()
        mock_nlp.return_value.ents = []
        mock_nlp.return_value.noun_chunks = []
        mock_spacy_load.return_value = mock_nlp
        
        agent = NLPAgent(
            agent_id="test_nlp_agent",
            memory_manager=memory_manager,
            llm_chain=mock_llm_chain
        )
        
        test_text = "This is a test sentence for NLP analysis."
        
        result = await agent.analyze_text(
            text=test_text,
            analysis_type="comprehensive"
        )
        
        assert "session_id" in result
        assert "analysis" in result
        assert "entities" in result["analysis"]
        assert "sentiment" in result["analysis"]

@pytest.mark.asyncio
async def test_agent_interaction(memory_manager, mock_llm_chain):
    """Test interaction between different agent types."""
    code_agent = CodeAgent(
        agent_id="test_code_agent",
        memory_manager=memory_manager,
        llm_chain=mock_llm_chain
    )
    
    nlp_agent = NLPAgent(
        agent_id="test_nlp_agent",
        memory_manager=memory_manager,
        llm_chain=mock_llm_chain
    )
    
    # Test code generation based on NLP analysis
    test_spec = "Create a function to process text data"
    nlp_analysis = await nlp_agent.analyze_text(test_spec)
    
    code_result = await code_agent.generate_code(
        spec={"description": test_spec, "nlp_analysis": nlp_analysis}
    )
    
    assert "session_id" in code_result
    assert "generation" in code_result

@pytest.mark.asyncio
async def test_error_handling(memory_manager, mock_llm_chain):
    """Test error handling in agents."""
    agent = CodeAgent(
        agent_id="test_code_agent",
        memory_manager=memory_manager,
        llm_chain=mock_llm_chain
    )
    
    # Test with invalid code
    result = await agent.analyze_code(
        code="invalid python code @@@@",
        context={"purpose": "testing"},
        requirements={"style_check": True}
    )
    
    assert "error" not in result  # Should handle errors gracefully
    assert "session_id" in result
    assert "analysis" in result
