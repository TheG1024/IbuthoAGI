"""
Critic agent implementation for solution evaluation and feedback generation.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..core.agent import Agent, AgentRole, AgentCapability
from ..utils.monitoring import MonitoringSystem

class CriticPrompt:
    """Prompts for evaluation and feedback."""
    
    EVALUATE_SOLUTION = PromptTemplate(
        input_variables=["solution", "requirements", "metrics"],
        template="""
        Evaluate the following solution against requirements and metrics:
        
        Solution: {solution}
        Requirements: {requirements}
        Metrics: {metrics}
        
        Provide:
        1. Requirements compliance analysis
        2. Performance evaluation
        3. Quality assessment
        4. Identified issues
        5. Improvement recommendations
        
        Format your response as a JSON object.
        """
    )
    
    GENERATE_FEEDBACK = PromptTemplate(
        input_variables=["evaluation", "context"],
        template="""
        Generate detailed feedback based on the evaluation:
        
        Evaluation: {evaluation}
        Context: {context}
        
        Provide:
        1. Key findings
        2. Strengths and weaknesses
        3. Actionable recommendations
        4. Priority improvements
        
        Format your response as a JSON object.
        """
    )

class CriticAgent(Agent):
    """Agent responsible for solution evaluation and feedback generation."""
    
    def __init__(
        self,
        agent_id: str,
        monitoring_system: MonitoringSystem,
        llm_chain: Optional[LLMChain] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.CRITIC,
            capabilities=[
                AgentCapability.SOLUTION_EVALUATION,
                AgentCapability.FEEDBACK_GENERATION
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.monitoring_system = monitoring_system
        self.llm_chain = llm_chain
        self.evaluations: Dict[str, Dict[str, Any]] = {}
    
    async def evaluate_solution(
        self,
        solution: Dict[str, Any],
        requirements: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a solution against requirements."""
        # Generate evaluation ID
        evaluation_id = f"eval_{datetime.now().timestamp()}"
        
        # Get metrics from monitoring system
        metrics = self.monitoring_system.get_metrics_summary()
        
        # Perform evaluation
        evaluation = await self._evaluate(
            solution,
            requirements,
            metrics
        )
        
        # Generate feedback
        feedback = await self._generate_feedback(
            evaluation,
            context
        )
        
        # Store evaluation
        self.evaluations[evaluation_id] = {
            "solution": solution,
            "requirements": requirements,
            "metrics": metrics,
            "evaluation": evaluation,
            "feedback": feedback,
            "timestamp": datetime.now(),
            "context": context
        }
        
        return {
            "evaluation_id": evaluation_id,
            "evaluation": evaluation,
            "feedback": feedback
        }
    
    async def _evaluate(
        self,
        solution: Dict[str, Any],
        requirements: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform detailed evaluation of solution."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            CriticPrompt.EVALUATE_SOLUTION,
            solution=json.dumps(solution),
            requirements=json.dumps(requirements),
            metrics=json.dumps(metrics)
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse evaluation response")
            return {}
    
    async def _generate_feedback(
        self,
        evaluation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate detailed feedback based on evaluation."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            CriticPrompt.GENERATE_FEEDBACK,
            evaluation=json.dumps(evaluation),
            context=json.dumps(context) if context else ""
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse feedback response")
            return {}
    
    def calculate_compliance_score(
        self,
        evaluation: Dict[str, Any]
    ) -> float:
        """Calculate requirements compliance score."""
        if not evaluation:
            return 0.0
            
        compliance_checks = evaluation.get("requirements_compliance", [])
        if not compliance_checks:
            return 0.0
            
        total_checks = len(compliance_checks)
        passed_checks = sum(
            1 for check in compliance_checks
            if check.get("status") == "passed"
        )
        
        return (passed_checks / total_checks) * 100 if total_checks > 0 else 0.0
    
    def calculate_quality_score(
        self,
        evaluation: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score."""
        if not evaluation:
            return 0.0
            
        quality_metrics = evaluation.get("quality_assessment", {})
        if not quality_metrics:
            return 0.0
            
        # Weight different quality aspects
        weights = {
            "reliability": 0.3,
            "performance": 0.3,
            "maintainability": 0.2,
            "security": 0.2
        }
        
        weighted_sum = sum(
            weights.get(metric, 0) * score
            for metric, score in quality_metrics.items()
            if metric in weights
        )
        
        return weighted_sum
    
    def get_evaluation_summary(
        self,
        evaluation_id: str
    ) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        evaluation_data = self.evaluations.get(evaluation_id, {})
        evaluation = evaluation_data.get("evaluation", {})
        feedback = evaluation_data.get("feedback", {})
        
        return {
            "timestamp": evaluation_data.get("timestamp"),
            "compliance_score": self.calculate_compliance_score(evaluation),
            "quality_score": self.calculate_quality_score(evaluation),
            "key_findings": feedback.get("key_findings", []),
            "priority_improvements": feedback.get("priority_improvements", []),
            "metrics_summary": evaluation_data.get("metrics", {})
        }
    
    def get_trend_analysis(
        self,
        solution_id: str,
        metric: str
    ) -> List[Dict[str, Any]]:
        """Get trend analysis for specific metric."""
        relevant_evaluations = [
            eval_data
            for eval_data in self.evaluations.values()
            if eval_data.get("solution", {}).get("id") == solution_id
        ]
        
        return [
            {
                "timestamp": eval_data["timestamp"],
                "value": eval_data.get("metrics", {}).get(metric),
                "evaluation_id": eval_id
            }
            for eval_id, eval_data in enumerate(relevant_evaluations)
            if metric in eval_data.get("metrics", {})
        ]
