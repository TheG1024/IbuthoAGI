"""
Specialized AI agents with distinct expertise and perspectives.
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

import openai
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class BaseAgent:
    def __init__(self, name: str, expertise: str, responsibilities: List[str]):
        self.name = name
        self.expertise = expertise
        self.responsibilities = responsibilities
        self.feedback_history = []
        
        # Initialize AI components
        self.llm = ChatOpenAI(temperature=0.7)
        self.embeddings = OpenAIEmbeddings()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Agent-specific prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "input_data"],
            template=f"""As a {self.name} with expertise in {self.expertise}, analyze the following:

Context: {{context}}
Input: {{input_data}}

Consider these responsibilities:
{chr(10).join(f'- {r}' for r in self.responsibilities)}

Provide detailed feedback addressing each responsibility."""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def provide_feedback(self, input_data: str, context: Optional[Dict] = None) -> str:
        """Generate AI-powered feedback based on agent's expertise."""
        try:
            # Generate embeddings for semantic understanding
            input_embedding = self.embeddings.embed_query(input_data)
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer(input_data)[0]
            
            # Get contextual embeddings
            context_str = str(context) if context else ""
            context_embedding = self.sentence_transformer.encode(context_str)
            
            # Generate feedback using LLM
            feedback = self.chain.run(
                context=context_str,
                input_data=input_data
            )
            
            # Document the analysis
            self.document_reasoning({
                "input_embedding": input_embedding,
                "sentiment": sentiment,
                "context_embedding": context_embedding,
                "feedback": feedback
            })
            
            return feedback
            
        except Exception as e:
            return f"Error generating feedback: {str(e)}"
    
    def document_reasoning(self, analysis: Dict[str, Any]) -> None:
        """Document the reasoning process with timestamps."""
        self.feedback_history.append({
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis
        })
    
    def challenge_assumptions(self, proposal: str) -> List[str]:
        """Identify and challenge assumptions using AI analysis."""
        try:
            # Create a prompt for assumption analysis
            assumption_prompt = PromptTemplate(
                input_variables=["proposal"],
                template="""Identify and challenge key assumptions in the following proposal:

Proposal: {proposal}

List each assumption and provide a critical analysis."""
            )
            
            assumption_chain = LLMChain(llm=self.llm, prompt=assumption_prompt)
            challenges = assumption_chain.run(proposal=proposal)
            
            return challenges.split("\n")
        except Exception as e:
            return [f"Error analyzing assumptions: {str(e)}"]

class TechnicalArchitect(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Technical Architect",
            expertise="System design and integration",
            responsibilities=[
                "Architecture design",
                "System integration",
                "Technical feasibility assessment",
                "Performance optimization"
            ]
        )
        # Add technical-specific analysis tools
        self.code_analyzer = pipeline("text-classification", model="microsoft/codebert-base")
    
    def analyze_technical_feasibility(self, solution_spec: str) -> Dict[str, Any]:
        """Analyze technical feasibility of proposed solutions."""
        try:
            # Analyze code/architecture patterns
            code_analysis = self.code_analyzer(solution_spec)
            
            # Generate technical assessment
            assessment_prompt = PromptTemplate(
                input_variables=["spec"],
                template="""Analyze the technical feasibility of:

{spec}

Provide assessment of:
1. Implementation complexity
2. Scalability concerns
3. Integration challenges
4. Performance implications"""
            )
            
            assessment_chain = LLMChain(llm=self.llm, prompt=assessment_prompt)
            assessment = assessment_chain.run(spec=solution_spec)
            
            return {
                "code_analysis": code_analysis,
                "feasibility_assessment": assessment
            }
        except Exception as e:
            return {"error": str(e)}

class CreativeDirector(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Creative Director",
            expertise="Innovation and user experience",
            responsibilities=[
                "Innovation strategy",
                "User experience design",
                "Creative problem-solving",
                "Design thinking facilitation"
            ]
        )
        self.idea_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    
    def generate_innovative_solutions(self, problem: str) -> List[str]:
        """Generate innovative solutions using AI."""
        try:
            # Generate multiple creative approaches
            solutions = self.idea_generator(
                f"Generate innovative solutions for: {problem}",
                max_length=200,
                num_return_sequences=3
            )
            
            return [solution["generated_text"] for solution in solutions]
        except Exception as e:
            return [f"Error generating solutions: {str(e)}"]

class DataScientist(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Data Scientist",
            expertise="Analytics and optimization",
            responsibilities=[
                "Data analysis",
                "Performance metrics",
                "Optimization strategies",
                "Quantitative assessment"
            ]
        )
        self.data_analyzer = pipeline("text2text-generation", model="google/flan-t5-base")

class DomainExpert(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Domain Expert",
            expertise="Subject matter knowledge",
            responsibilities=[
                "Domain-specific guidance",
                "Best practices",
                "Industry standards",
                "Technical requirements"
            ]
        )

class QualityAssurance(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Quality Assurance",
            expertise="Testing and validation",
            responsibilities=[
                "Quality control",
                "Testing strategies",
                "Validation protocols",
                "Performance monitoring"
            ]
        )

class RiskAnalyst(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Risk Analyst",
            expertise="Security and compliance",
            responsibilities=[
                "Risk assessment",
                "Compliance checking",
                "Security analysis",
                "Mitigation strategies"
            ]
        )

class UserAdvocate(BaseAgent):
    def __init__(self):
        super().__init__(
            name="User Advocate",
            expertise="Accessibility and usability",
            responsibilities=[
                "User needs assessment",
                "Accessibility requirements",
                "Usability testing",
                "User feedback integration"
            ]
        )

class EthicsOfficer(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Ethics Officer",
            expertise="Responsible AI practices",
            responsibilities=[
                "Ethical guidelines",
                "Bias detection",
                "Fairness assessment",
                "Responsible AI principles"
            ]
        )

class AgentCollaboration:
    def __init__(self):
        self.agents = {
            "technical": TechnicalArchitect(),
            "creative": CreativeDirector(),
            "data": DataScientist(),
            "domain": DomainExpert(),
            "qa": QualityAssurance(),
            "risk": RiskAnalyst(),
            "user": UserAdvocate(),
            "ethics": EthicsOfficer()
        }
        
        # Initialize cross-agent communication
        self.communication_embeddings = OpenAIEmbeddings()
    
    def get_collaborative_feedback(
        self,
        input_data: str,
        context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Gather AI-powered feedback from all relevant agents."""
        feedback = {}
        
        # Get input embedding for relevance matching
        input_embedding = self.communication_embeddings.embed_query(input_data)
        
        for agent_name, agent in self.agents.items():
            # Calculate relevance score
            agent_embedding = self.communication_embeddings.embed_query(agent.expertise)
            relevance = np.dot(input_embedding, agent_embedding)
            
            # Get feedback if agent is relevant
            if relevance > 0.5:  # Relevance threshold
                feedback[agent_name] = agent.provide_feedback(input_data, context)
        
        return feedback
    
    def synthesize_feedback(
        self,
        feedback_collection: Dict[str, str]
    ) -> Dict[str, Any]:
        """Synthesize feedback from multiple agents into actionable insights."""
        try:
            synthesis_prompt = PromptTemplate(
                input_variables=["feedback"],
                template="""Synthesize the following feedback from multiple experts:

{feedback}

Provide:
1. Key insights
2. Common themes
3. Potential conflicts
4. Recommended actions"""
            )
            
            synthesis_chain = LLMChain(
                llm=ChatOpenAI(temperature=0.3),
                prompt=synthesis_prompt
            )
            
            synthesis = synthesis_chain.run(
                feedback="\n\n".join(
                    f"{agent}: {feedback}"
                    for agent, feedback in feedback_collection.items()
                )
            )
            
            return {
                "synthesis": synthesis,
                "source_feedback": feedback_collection
            }
        except Exception as e:
            return {"error": str(e)}
