"""
Code-focused agent implementation for software development tasks.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import ast
import astor
import black

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from ..core.agent import Agent, AgentRole, AgentCapability
from ..utils.memory_manager import MemoryManager

class CodePrompt:
    """Prompts for code-related operations."""
    
    CODE_ANALYSIS = PromptTemplate(
        input_variables=["code", "context", "requirements"],
        template="""
        Analyze the following code:
        
        Code: {code}
        Context: {context}
        Requirements: {requirements}
        
        Provide:
        1. Code structure analysis
        2. Quality assessment
        3. Potential issues
        4. Optimization opportunities
        5. Security concerns
        
        Format your response as a JSON object.
        """
    )
    
    CODE_GENERATION = PromptTemplate(
        input_variables=["spec", "context", "constraints"],
        template="""
        Generate code based on the following specification:
        
        Specification: {spec}
        Context: {context}
        Constraints: {constraints}
        
        Provide:
        1. Implementation code
        2. Documentation
        3. Test cases
        4. Usage examples
        5. Error handling
        
        Format your response as a JSON object.
        """
    )

class CodeAgent(Agent):
    """Agent specialized in code analysis, generation, and optimization."""
    
    def __init__(
        self,
        agent_id: str,
        memory_manager: MemoryManager,
        llm_chain: Optional[LLMChain] = None,
        embeddings_model: Optional[OpenAIEmbeddings] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.CODE_GENERATION
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.memory_manager = memory_manager
        self.llm_chain = llm_chain
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.code_store = FAISS.from_texts(
            ["Initial code store"],
            embedding=self.embeddings
        )
        self.code_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze code for quality, issues, and improvements."""
        # Generate session ID
        session_id = f"code_analysis_{datetime.now().timestamp()}"
        
        # Perform analysis
        analysis = await self._analyze_code(
            code,
            context,
            requirements
        )
        
        # Store session
        self.code_sessions[session_id] = {
            "code": code,
            "context": context,
            "requirements": requirements,
            "analysis": analysis,
            "timestamp": datetime.now(),
            "type": "analysis"
        }
        
        return {
            "session_id": session_id,
            "analysis": analysis
        }
    
    async def generate_code(
        self,
        spec: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate code based on specifications."""
        # Generate session ID
        session_id = f"code_gen_{datetime.now().timestamp()}"
        
        # Generate code
        generation = await self._generate_code(
            spec,
            context,
            constraints
        )
        
        # Format code
        formatted_code = await self._format_code(
            generation.get("implementation", "")
        )
        generation["implementation"] = formatted_code
        
        # Store session
        self.code_sessions[session_id] = {
            "spec": spec,
            "context": context,
            "constraints": constraints,
            "generation": generation,
            "timestamp": datetime.now(),
            "type": "generation"
        }
        
        return {
            "session_id": session_id,
            "generation": generation
        }
    
    async def optimize_code(
        self,
        code: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize code for performance and quality."""
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Apply optimizations
            optimized_tree = await self._apply_optimizations(tree)
            
            # Generate optimized code
            optimized_code = astor.to_source(optimized_tree)
            
            # Format code
            formatted_code = await self._format_code(optimized_code)
            
            return {
                "optimized_code": formatted_code,
                "metrics": metrics
            }
        except Exception as e:
            self.logger.error(f"Code optimization failed: {str(e)}")
            return {
                "error": str(e),
                "original_code": code
            }
    
    async def _analyze_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]],
        requirements: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform detailed code analysis."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            CodePrompt.CODE_ANALYSIS,
            code=code,
            context=json.dumps(context) if context else "",
            requirements=json.dumps(requirements) if requirements else ""
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse code analysis response")
            return {}
    
    async def _generate_code(
        self,
        spec: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate code based on specifications."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            CodePrompt.CODE_GENERATION,
            spec=json.dumps(spec),
            context=json.dumps(context) if context else "",
            constraints=json.dumps(constraints) if constraints else ""
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse code generation response")
            return {}
    
    async def _apply_optimizations(self, tree: ast.AST) -> ast.AST:
        """Apply code optimizations to AST."""
        # This would contain various optimization strategies
        # For now, we'll return the original tree
        return tree
    
    async def _format_code(self, code: str) -> str:
        """Format code using black."""
        try:
            return black.format_str(
                code,
                mode=black.FileMode()
            )
        except Exception as e:
            self.logger.error(f"Code formatting failed: {str(e)}")
            return code
    
    def get_session_details(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about a code session."""
        session = self.code_sessions.get(session_id, {})
        return {
            "timestamp": session.get("timestamp"),
            "type": session.get("type"),
            "analysis": session.get("analysis"),
            "generation": session.get("generation")
        }
    
    def get_similar_code(
        self,
        code: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar code examples from the store."""
        results = self.code_store.similarity_search(
            code,
            k=limit
        )
        
        return [
            {
                "content": result.page_content,
                "metadata": result.metadata
            }
            for result in results
        ]
