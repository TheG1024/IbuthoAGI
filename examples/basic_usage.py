"""
Example usage of the IbuthoAGI prompt engineering system.
"""
from src.agents.specialized_agents import AgentCollaboration
from src.core.chain_prompts import ChainPrompts
from src.workflow.evaluation_system import WorkflowManager, EvaluationSystem

def main():
    # Initialize the system components
    agents = AgentCollaboration()
    prompts = ChainPrompts()
    workflow = WorkflowManager()
    evaluator = EvaluationSystem()
    
    # Example problem statement
    problem = {
        "title": "Design a sustainable smart home system",
        "description": """
        Create an AI-powered home automation system that optimizes energy usage,
        enhances comfort, and promotes sustainable living practices while ensuring
        user privacy and data security.
        """,
        "constraints": [
            "Must be cost-effective",
            "Privacy-preserving",
            "Energy efficient",
            "User-friendly interface"
        ]
    }
    
    # 1. Problem Analysis Stage
    analysis_prompt = prompts.get_prompt(
        "problem_definition",
        problem_statement=problem["description"]
    )
    
    # Get feedback from relevant agents
    analysis_feedback = agents.get_collaborative_feedback(
        analysis_prompt,
        context={"stage": "problem_analysis"}
    )
    
    # 2. Solution Ideation Stage
    ideation_prompt = prompts.get_prompt(
        "solution_ideation",
        problem_analysis=analysis_feedback
    )
    
    solution_feedback = agents.get_collaborative_feedback(
        ideation_prompt,
        context={"stage": "solution_ideation"}
    )
    
    # 3. Technical Assessment
    technical_prompt = prompts.get_prompt(
        "technical_planning",
        selected_solution=solution_feedback
    )
    
    technical_feedback = agents.get_collaborative_feedback(
        technical_prompt,
        context={"stage": "technical_assessment"}
    )
    
    # 4. Evaluate the solution
    evaluation_results = evaluator.evaluate_solution({
        "analysis": analysis_feedback,
        "solution": solution_feedback,
        "technical": technical_feedback
    })
    
    # Print results
    print("\nProblem Analysis Feedback:")
    for agent, feedback in analysis_feedback.items():
        print(f"\n{agent.title()} Agent Feedback:")
        print(feedback)
    
    print("\nSolution Evaluation Results:")
    for criterion, score in evaluation_results.items():
        print(f"{criterion}: {score}")

if __name__ == "__main__":
    main()
