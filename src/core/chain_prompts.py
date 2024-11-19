"""
Core chain prompts for sequential reasoning in the prompt engineering system.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import openai
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptStage(str, Enum):
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"

@dataclass
class PromptTemplate:
    template: str
    stage: PromptStage
    variables: list[str]
    description: str

class ChainPrompts:
    def __init__(self):
        # Initialize AI components
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        self.prompts = self._initialize_prompts()
        self.chains = self._initialize_chains()
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize the prompt templates with AI-optimized content."""
        return {
            # Problem Analysis
            "problem_definition": PromptTemplate(
                template="""System: You are an expert problem analyst. Your task is to analyze the following problem comprehensively.

Problem Statement:
{problem_statement}

Analyze the problem considering:
1. Key objectives and goals
2. Constraints and limitations
3. Success criteria and metrics
4. Potential challenges and risks
5. Stakeholder requirements
6. Technical considerations
7. Resource implications

Provide a detailed analysis addressing each point.""",
                stage=PromptStage.ANALYSIS,
                variables=["problem_statement"],
                description="Comprehensive problem analysis"
            ),
            
            "context_gathering": PromptTemplate(
                template="""System: You are a context analysis specialist. Gather and analyze relevant context for the problem.

Problem:
{problem_statement}

Consider and analyze:
1. Domain knowledge requirements
2. Similar existing solutions
3. Industry best practices
4. Available resources and constraints
5. Stakeholder ecosystem
6. Technical environment
7. Market conditions

Provide detailed context analysis for each aspect.""",
                stage=PromptStage.ANALYSIS,
                variables=["problem_statement"],
                description="Context gathering and analysis"
            ),
            
            # Solution Design
            "solution_ideation": PromptTemplate(
                template="""System: You are an innovative solution architect. Generate creative and effective solutions.

Problem Analysis:
{problem_analysis}

Generate solutions considering:
1. Technical feasibility
2. Innovation potential
3. Resource requirements
4. Implementation timeline
5. Scalability aspects
6. Integration requirements
7. Performance implications

Present multiple solution approaches with detailed analysis.""",
                stage=PromptStage.DESIGN,
                variables=["problem_analysis"],
                description="Solution ideation and design"
            ),
            
            # Implementation Strategy
            "technical_planning": PromptTemplate(
                template="""System: You are a technical implementation specialist. Create a detailed implementation plan.

Selected Solution:
{selected_solution}

Provide a comprehensive plan including:
1. Architecture components
2. System integration points
3. Development phases
4. Technical requirements
5. Resource allocation
6. Timeline estimates
7. Risk mitigation strategies

Detail each aspect of the implementation plan.""",
                stage=PromptStage.IMPLEMENTATION,
                variables=["selected_solution"],
                description="Technical implementation planning"
            ),
            
            # Validation
            "quality_assessment": PromptTemplate(
                template="""System: You are a quality assurance expert. Evaluate the proposed solution thoroughly.

Solution Details:
{solution_details}

Evaluate against these criteria:
1. Technical robustness
2. Performance metrics
3. Security considerations
4. Scalability aspects
5. User experience
6. Maintenance requirements
7. Compliance standards

Provide a detailed assessment with recommendations.""",
                stage=PromptStage.VALIDATION,
                variables=["solution_details"],
                description="Quality and validation assessment"
            )
        }
    
    def _initialize_chains(self) -> Dict[str, LLMChain]:
        """Initialize LangChain chains for each prompt template."""
        chains = {}
        for prompt_key, prompt_data in self.prompts.items():
            # Create message prompts
            system_message_prompt = SystemMessagePromptTemplate.from_template(
                "You are an AI assistant specialized in " + prompt_data.description
            )
            
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                prompt_data.template
            )
            
            chat_prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                human_message_prompt
            ])
            
            # Create chain
            chains[prompt_key] = LLMChain(
                llm=self.llm,
                prompt=chat_prompt,
                verbose=True
            )
        
        return chains
    
    def get_prompt(self, prompt_key: str, **kwargs: Any) -> str:
        """Retrieve and format a specific prompt template with AI-generated content."""
        if prompt_key not in self.prompts:
            raise KeyError(f"Prompt '{prompt_key}' not found")
        
        try:
            # Get the chain for this prompt
            chain = self.chains[prompt_key]
            
            # Generate response
            response = chain.run(**kwargs)
            
            # Add to conversation memory
            self.conversation.predict(input=response)
            
            return response
        
        except Exception as e:
            return f"Error generating prompt: {str(e)}"
    
    def get_prompts_by_stage(self, stage: PromptStage) -> Dict[str, PromptTemplate]:
        """Get all prompts for a specific stage."""
        return {
            k: v for k, v in self.prompts.items()
            if v.stage == stage
        }
    
    def generate_custom_prompt(
        self,
        context: str,
        stage: PromptStage,
        specific_requirements: Optional[Dict] = None
    ) -> str:
        """Generate a custom prompt based on context and requirements."""
        try:
            custom_prompt = PromptTemplate(
                input_variables=["context", "requirements"],
                template="""Generate a specialized prompt for the following context:

Context: {context}
Stage: {stage}
Requirements: {requirements}

The prompt should be:
1. Specific to the context
2. Aligned with the stage
3. Addressing all requirements
4. Following best practices for AI interaction

Generate a detailed prompt template."""
            )
            
            chain = LLMChain(llm=self.llm, prompt=custom_prompt)
            
            return chain.run(
                context=context,
                stage=stage.value,
                requirements=str(specific_requirements)
            )
        
        except Exception as e:
            return f"Error generating custom prompt: {str(e)}"
