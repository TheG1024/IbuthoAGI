"""
Innovator agent implementation for creative solution generation and optimization.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import random

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from ..core.agent import Agent, AgentRole, AgentCapability
from ..utils.memory_manager import MemoryManager

class InnovationPrompt:
    """Prompts for innovation and optimization."""
    
    GENERATE_IDEAS = PromptTemplate(
        input_variables=["problem", "context", "constraints", "existing_solutions"],
        template="""
        Generate innovative solutions for the following problem:
        
        Problem: {problem}
        Context: {context}
        Constraints: {constraints}
        Existing Solutions: {existing_solutions}
        
        Provide:
        1. Novel solution approaches
        2. Creative adaptations
        3. Unique combinations
        4. Breakthrough opportunities
        5. Implementation concepts
        
        Format your response as a JSON object.
        """
    )
    
    OPTIMIZE_SOLUTION = PromptTemplate(
        input_variables=["solution", "metrics", "constraints", "objectives"],
        template="""
        Optimize the following solution:
        
        Solution: {solution}
        Current Metrics: {metrics}
        Constraints: {constraints}
        Optimization Objectives: {objectives}
        
        Provide:
        1. Optimization strategies
        2. Performance improvements
        3. Resource optimizations
        4. Trade-off analysis
        5. Implementation recommendations
        
        Format your response as a JSON object.
        """
    )

class InnovatorAgent(Agent):
    """Agent responsible for creative solution generation and optimization."""
    
    def __init__(
        self,
        agent_id: str,
        memory_manager: MemoryManager,
        llm_chain: Optional[LLMChain] = None,
        embeddings_model: Optional[OpenAIEmbeddings] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.INNOVATOR,
            capabilities=[
                AgentCapability.SOLUTION_SYNTHESIS,
                AgentCapability.OPTIMIZATION
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.memory_manager = memory_manager
        self.llm_chain = llm_chain
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.innovation_store = FAISS.from_texts(
            ["Initial innovation store"],
            embedding=self.embeddings
        )
        self.innovations: Dict[str, Dict[str, Any]] = {}
    
    async def generate_innovations(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate innovative solutions for a problem."""
        # Generate innovation ID
        innovation_id = f"innov_{datetime.now().timestamp()}"
        
        # Get existing solutions from memory
        existing_solutions = await self._get_existing_solutions(problem)
        
        # Generate ideas
        ideas = await self._generate_ideas(
            problem,
            context,
            constraints,
            existing_solutions
        )
        
        # Store innovation
        self.innovations[innovation_id] = {
            "problem": problem,
            "context": context,
            "constraints": constraints,
            "ideas": ideas,
            "timestamp": datetime.now(),
            "status": "generated"
        }
        
        # Store in innovation store
        self._store_innovation(innovation_id, ideas)
        
        return {
            "innovation_id": innovation_id,
            "ideas": ideas
        }
    
    async def optimize_solution(
        self,
        solution: Dict[str, Any],
        metrics: Dict[str, Any],
        objectives: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize an existing solution."""
        # Generate optimization ID
        optimization_id = f"opt_{datetime.now().timestamp()}"
        
        # Generate optimization strategies
        optimization = await self._generate_optimization(
            solution,
            metrics,
            objectives,
            constraints
        )
        
        # Store optimization
        self.innovations[optimization_id] = {
            "solution": solution,
            "metrics": metrics,
            "objectives": objectives,
            "constraints": constraints,
            "optimization": optimization,
            "timestamp": datetime.now(),
            "status": "optimized"
        }
        
        return {
            "optimization_id": optimization_id,
            "optimization": optimization
        }
    
    async def _get_existing_solutions(
        self,
        problem: str
    ) -> List[Dict[str, Any]]:
        """Retrieve existing solutions from memory."""
        # Search memory for relevant solutions
        memory_results = await self.memory_manager.search_memory(
            [problem],
            memory_type="solutions"
        )
        
        # Search innovation store
        similar_innovations = self.innovation_store.similarity_search(
            problem,
            k=5
        )
        
        # Combine results
        solutions = []
        if memory_results:
            solutions.extend(memory_results)
        if similar_innovations:
            solutions.extend([
                {"source": "innovation_store", "content": result.page_content}
                for result in similar_innovations
            ])
        
        return solutions
    
    async def _generate_ideas(
        self,
        problem: str,
        context: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
        existing_solutions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate innovative ideas."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            InnovationPrompt.GENERATE_IDEAS,
            problem=problem,
            context=json.dumps(context) if context else "",
            constraints=json.dumps(constraints) if constraints else "",
            existing_solutions=json.dumps(existing_solutions)
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse ideas generation response")
            return {}
    
    async def _generate_optimization(
        self,
        solution: Dict[str, Any],
        metrics: Dict[str, Any],
        objectives: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate optimization strategies."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            InnovationPrompt.OPTIMIZE_SOLUTION,
            solution=json.dumps(solution),
            metrics=json.dumps(metrics),
            constraints=json.dumps(constraints) if constraints else "",
            objectives=json.dumps(objectives)
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse optimization response")
            return {}
    
    def _store_innovation(
        self,
        innovation_id: str,
        ideas: Dict[str, Any]
    ) -> None:
        """Store innovation in vector store."""
        self.innovation_store.add_texts(
            [json.dumps(ideas)],
            metadatas=[{"innovation_id": innovation_id}]
        )
    
    def get_innovation_details(
        self,
        innovation_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about an innovation."""
        innovation = self.innovations.get(innovation_id, {})
        return {
            "timestamp": innovation.get("timestamp"),
            "status": innovation.get("status"),
            "problem": innovation.get("problem"),
            "ideas": innovation.get("ideas", {}),
            "optimization": innovation.get("optimization", {})
        }
    
    def get_similar_innovations(
        self,
        problem: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar innovations from the store."""
        results = self.innovation_store.similarity_search(
            problem,
            k=limit
        )
        
        return [
            {
                "content": result.page_content,
                "metadata": result.metadata
            }
            for result in results
        ]
    
    def combine_ideas(
        self,
        ideas: List[Dict[str, Any]],
        combination_strategy: str = "random"
    ) -> Dict[str, Any]:
        """Combine multiple ideas into a new innovation."""
        if not ideas:
            return {}
            
        if combination_strategy == "random":
            # Randomly select elements from different ideas
            combined = {}
            for idea in ideas:
                key = random.choice(list(idea.keys()))
                combined[key] = idea[key]
            return combined
        
        # Default to taking the best elements from each idea
        return {
            key: max(
                (idea.get(key) for idea in ideas if key in idea),
                key=lambda x: len(str(x)) if x else 0
            )
            for key in set().union(*(idea.keys() for idea in ideas))
        }
