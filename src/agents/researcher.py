"""
Researcher agent implementation for information gathering and analysis.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from ..core.agent import Agent, AgentRole, AgentCapability
from ..utils.memory_manager import MemoryManager

class ResearchPrompt:
    """Prompts for research operations."""
    
    ANALYZE_QUERY = PromptTemplate(
        input_variables=["query", "context"],
        template="""
        Analyze the following research query in the given context:
        
        Query: {query}
        Context: {context}
        
        Provide:
        1. Key information requirements
        2. Potential information sources
        3. Research approach and methodology
        4. Expected deliverables
        
        Format your response as a JSON object.
        """
    )
    
    SYNTHESIZE_FINDINGS = PromptTemplate(
        input_variables=["findings", "query"],
        template="""
        Synthesize the following research findings for the given query:
        
        Query: {query}
        Findings: {findings}
        
        Provide:
        1. Key insights and patterns
        2. Supporting evidence
        3. Knowledge gaps
        4. Recommendations
        
        Format your response as a JSON object.
        """
    )

class ResearcherAgent(Agent):
    """Agent responsible for information gathering and analysis."""
    
    def __init__(
        self,
        agent_id: str,
        memory_manager: MemoryManager,
        llm_chain: Optional[LLMChain] = None,
        embeddings_model: Optional[OpenAIEmbeddings] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.RESEARCHER,
            capabilities=[
                AgentCapability.INFORMATION_GATHERING,
                AgentCapability.SOLUTION_SYNTHESIS
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.memory_manager = memory_manager
        self.llm_chain = llm_chain
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.knowledge_base = FAISS.from_texts(
            ["Initial knowledge base"],
            embedding=self.embeddings
        )
        self.active_research: Dict[str, Dict[str, Any]] = {}
    
    async def conduct_research(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Conduct research on a given query."""
        # Create research session
        research_id = f"research_{datetime.now().timestamp()}"
        
        # Analyze query
        analysis = await self._analyze_query(query, context)
        
        # Initialize research session
        self.active_research[research_id] = {
            "query": query,
            "context": context,
            "analysis": analysis,
            "findings": [],
            "status": "in_progress",
            "started_at": datetime.now()
        }
        
        # Gather information
        findings = await self._gather_information(
            research_id,
            analysis
        )
        
        # Synthesize findings
        synthesis = await self._synthesize_findings(
            research_id,
            findings
        )
        
        # Store in knowledge base
        await self._store_findings(research_id, synthesis)
        
        return {
            "research_id": research_id,
            "analysis": analysis,
            "findings": findings,
            "synthesis": synthesis
        }
    
    async def _analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze research query."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            ResearchPrompt.ANALYZE_QUERY,
            query=query,
            context=json.dumps(context) if context else ""
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse query analysis response")
            return {}
    
    async def _gather_information(
        self,
        research_id: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Gather information based on analysis."""
        findings = []
        
        # Search memory for relevant information
        memory_results = await self.memory_manager.search_memory(
            analysis.get("key_requirements", [])
        )
        if memory_results:
            findings.extend(memory_results)
        
        # Search knowledge base
        kb_results = self.knowledge_base.similarity_search(
            str(analysis.get("key_requirements", [])),
            k=5
        )
        if kb_results:
            findings.extend([
                {"source": "knowledge_base", "content": result.page_content}
                for result in kb_results
            ])
        
        # Update research session
        research = self.active_research.get(research_id)
        if research:
            research["findings"].extend(findings)
        
        return findings
    
    async def _synthesize_findings(
        self,
        research_id: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize research findings."""
        if not self.llm_chain:
            return {}
            
        research = self.active_research.get(research_id)
        if not research:
            return {}
            
        response = await self.llm_chain.arun(
            ResearchPrompt.SYNTHESIZE_FINDINGS,
            findings=json.dumps(findings),
            query=research["query"]
        )
        
        try:
            synthesis = json.loads(response)
            research["status"] = "completed"
            research["completed_at"] = datetime.now()
            return synthesis
        except json.JSONDecodeError:
            self.logger.error("Failed to parse findings synthesis response")
            return {}
    
    async def _store_findings(
        self,
        research_id: str,
        synthesis: Dict[str, Any]
    ) -> None:
        """Store research findings in knowledge base."""
        # Store in memory manager
        await self.memory_manager.store_memory(
            content=synthesis,
            memory_type="research_findings",
            metadata={
                "research_id": research_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Update knowledge base
        self.knowledge_base.add_texts(
            [json.dumps(synthesis)],
            metadatas=[{"research_id": research_id}]
        )
    
    def get_research_status(self, research_id: str) -> Dict[str, Any]:
        """Get research status."""
        research = self.active_research.get(research_id, {})
        return {
            "status": research.get("status"),
            "query": research.get("query"),
            "started_at": research.get("started_at"),
            "completed_at": research.get("completed_at"),
            "findings_count": len(research.get("findings", [])),
            "analysis": research.get("analysis")
        }
