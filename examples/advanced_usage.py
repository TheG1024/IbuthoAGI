"""
Advanced usage example demonstrating the full capabilities of IbuthoAGI.
"""
import asyncio
import logging
from typing import Dict, Any

from src.agents import (
    CoordinatorAgent,
    ResearcherAgent,
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
    InnovatorAgent,
    CodeAgent,
    DataScienceAgent,
    NLPAgent
)
from src.utils.memory_manager import MemoryManager
from src.core.agent_manager import AgentManager
from src.utils.monitoring import setup_monitoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def solve_complex_problem(
    task_description: str,
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Solve a complex problem using all available agents.
    
    Args:
        task_description: Detailed description of the task
        requirements: Dictionary of requirements and constraints
    
    Returns:
        Dictionary containing the solution and related artifacts
    """
    # Initialize components
    memory_manager = MemoryManager()
    
    # Initialize specialized agents
    code_agent = CodeAgent("code", memory_manager)
    ds_agent = DataScienceAgent("data_science", memory_manager)
    nlp_agent = NLPAgent("nlp", memory_manager)
    
    # Initialize core agents
    coordinator = CoordinatorAgent("coordinator", memory_manager)
    researcher = ResearcherAgent("researcher", memory_manager)
    planner = PlannerAgent("planner", memory_manager)
    executor = ExecutorAgent("executor", memory_manager)
    critic = CriticAgent("critic", memory_manager)
    innovator = InnovatorAgent("innovator", memory_manager)
    
    # Initialize agent manager
    agent_manager = AgentManager([
        coordinator, researcher, planner,
        executor, critic, innovator,
        code_agent, ds_agent, nlp_agent
    ])
    
    # Setup monitoring
    setup_monitoring()
    
    try:
        # 1. Initial Analysis
        logger.info("Starting initial analysis...")
        nlp_analysis = await nlp_agent.analyze_text(
            text=task_description,
            analysis_type="task_analysis",
            requirements=requirements
        )
        
        # 2. Research Phase
        logger.info("Conducting research...")
        research_results = await researcher.research_task({
            "task": task_description,
            "nlp_analysis": nlp_analysis,
            "requirements": requirements
        })
        
        # 3. Planning Phase
        logger.info("Creating solution plan...")
        solution_plan = await planner.create_plan({
            "task": task_description,
            "research": research_results,
            "requirements": requirements
        })
        
        # 4. Innovation Phase
        logger.info("Generating innovative solutions...")
        innovative_solutions = await innovator.generate_solutions({
            "task": task_description,
            "plan": solution_plan,
            "research": research_results
        })
        
        # 5. Code Generation (if needed)
        if "code_generation" in requirements:
            logger.info("Generating code solutions...")
            code_solution = await code_agent.generate_code(
                spec={
                    "task": task_description,
                    "plan": solution_plan,
                    "innovations": innovative_solutions
                }
            )
        
        # 6. Data Analysis (if needed)
        if "data_analysis" in requirements:
            logger.info("Performing data analysis...")
            data_analysis = await ds_agent.analyze_data(
                data=requirements.get("data"),
                analysis_requirements=requirements.get("analysis_requirements")
            )
        
        # 7. Execution Phase
        logger.info("Executing solution...")
        execution_result = await executor.execute_plan({
            "plan": solution_plan,
            "code_solution": code_solution if "code_generation" in requirements else None,
            "data_analysis": data_analysis if "data_analysis" in requirements else None
        })
        
        # 8. Evaluation Phase
        logger.info("Evaluating solution...")
        evaluation = await critic.evaluate_solution({
            "task": task_description,
            "requirements": requirements,
            "execution_result": execution_result,
            "innovative_solutions": innovative_solutions
        })
        
        # 9. Final Integration
        logger.info("Integrating results...")
        final_result = await coordinator.integrate_results({
            "task": task_description,
            "research": research_results,
            "plan": solution_plan,
            "execution": execution_result,
            "evaluation": evaluation,
            "code_solution": code_solution if "code_generation" in requirements else None,
            "data_analysis": data_analysis if "data_analysis" in requirements else None
        })
        
        return final_result
        
    except Exception as e:
        logger.error(f"Error solving problem: {str(e)}")
        raise

async def main():
    # Example task
    task_description = """
    Create a machine learning model to predict customer churn based on historical data.
    The solution should include:
    1. Data preprocessing and analysis
    2. Feature engineering
    3. Model selection and training
    4. Evaluation metrics
    5. API endpoint for predictions
    """
    
    requirements = {
        "code_generation": True,
        "data_analysis": True,
        "analysis_requirements": {
            "correlation_analysis": True,
            "feature_importance": True
        },
        "model_requirements": {
            "type": "classification",
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "cross_validation": True
        },
        "api_requirements": {
            "framework": "fastapi",
            "authentication": True,
            "rate_limiting": True
        }
    }
    
    try:
        result = await solve_complex_problem(
            task_description=task_description,
            requirements=requirements
        )
        
        logger.info("Problem solved successfully!")
        logger.info(f"Result: {result}")
        
    except Exception as e:
        logger.error(f"Failed to solve problem: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
