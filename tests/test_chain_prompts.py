"""
Tests for the chain prompts module.
"""
import pytest
from src.core.chain_prompts import ChainPrompts

def test_prompt_initialization():
    prompts = ChainPrompts()
    assert prompts.prompts is not None
    assert len(prompts.prompts) > 0

def test_get_prompt():
    prompts = ChainPrompts()
    problem_statement = "Design a sustainable smart home system"
    
    # Test problem definition prompt
    prompt = prompts.get_prompt("problem_definition", problem_statement=problem_statement)
    assert prompt is not None
    assert problem_statement in prompt
    
    # Test invalid prompt key
    with pytest.raises(KeyError):
        prompts.get_prompt("nonexistent_prompt")

def test_prompt_stages():
    prompts = ChainPrompts()
    
    # Check if all prompts have valid stages
    valid_stages = {"analysis", "design", "implementation", "validation"}
    for prompt_data in prompts.prompts.values():
        assert prompt_data["stage"] in valid_stages
