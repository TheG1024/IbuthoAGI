"""
NLP agent implementation for text analysis and processing tasks.
"""
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from ..core.agent import Agent, AgentRole, AgentCapability
from ..utils.memory_manager import MemoryManager

class NLPPrompt:
    """Prompts for NLP operations."""
    
    TEXT_ANALYSIS = PromptTemplate(
        input_variables=["text", "analysis_type", "requirements"],
        template="""
        Analyze the following text:
        
        Text: {text}
        Analysis Type: {analysis_type}
        Requirements: {requirements}
        
        Provide:
        1. Main themes/topics
        2. Key entities
        3. Sentiment analysis
        4. Style analysis
        5. Recommendations
        
        Format your response as a JSON object.
        """
    )
    
    GENERATION_PROMPT = PromptTemplate(
        input_variables=["context", "style", "requirements"],
        template="""
        Generate text based on:
        
        Context: {context}
        Style: {style}
        Requirements: {requirements}
        
        Provide:
        1. Generated text
        2. Style elements used
        3. Key themes
        4. Target audience
        5. Tone analysis
        
        Format your response as a JSON object.
        """
    )

class NLPAgent(Agent):
    """Agent specialized in natural language processing."""
    
    def __init__(
        self,
        agent_id: str,
        memory_manager: MemoryManager,
        llm_chain: Optional[LLMChain] = None,
        embeddings_model: Optional[OpenAIEmbeddings] = None,
        spacy_model: str = "en_core_web_sm"
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability.TEXT_ANALYSIS,
                AgentCapability.TEXT_GENERATION
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.memory_manager = memory_manager
        self.llm_chain = llm_chain
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        
        # Load models
        self.nlp = spacy.load(spacy_model)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")
        self.ner = pipeline("ner")
        
        # Initialize session storage
        self.analysis_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "comprehensive",
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive text analysis."""
        # Generate session ID
        session_id = f"text_analysis_{datetime.now().timestamp()}"
        
        # Basic analysis
        doc = self.nlp(text)
        
        # Entity recognition
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Key phrases and noun chunks
        noun_chunks = [
            {
                "text": chunk.text,
                "root": chunk.root.text,
                "root_dep": chunk.root.dep_
            }
            for chunk in doc.noun_chunks
        ]
        
        # Dependency parsing
        dependencies = [
            {
                "text": token.text,
                "dep": token.dep_,
                "head": token.head.text
            }
            for token in doc
        ]
        
        # Summarization for longer texts
        summary = ""
        if len(text.split()) > 100:
            summary = self.summarizer(
                text,
                max_length=130,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
        
        # Store analysis results
        analysis_results = {
            "entities": entities,
            "sentiment": sentiment,
            "noun_chunks": noun_chunks,
            "dependencies": dependencies,
            "summary": summary,
            "requirements_analysis": await self._analyze_requirements(
                text,
                analysis_type,
                requirements
            )
        }
        
        # Store session
        self.analysis_sessions[session_id] = {
            "timestamp": datetime.now(),
            "text_length": len(text),
            "analysis_type": analysis_type,
            "requirements": requirements,
            "results": analysis_results
        }
        
        return {
            "session_id": session_id,
            "analysis": analysis_results
        }
    
    async def generate_text(
        self,
        context: str,
        style: Optional[Dict[str, Any]] = None,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text based on context and style requirements."""
        if not self.llm_chain:
            return {"error": "LLM chain not initialized"}
        
        try:
            response = await self.llm_chain.arun(
                NLPPrompt.GENERATION_PROMPT,
                context=context,
                style=json.dumps(style) if style else "",
                requirements=json.dumps(requirements) if requirements else ""
            )
            
            try:
                generation_results = json.loads(response)
                
                # Analyze generated text
                analysis = await self.analyze_text(
                    generation_results.get("generated_text", ""),
                    analysis_type="style_check"
                )
                
                generation_results["analysis"] = analysis
                
                return generation_results
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse text generation response")
                return {
                    "error": "Failed to parse generation results",
                    "raw_response": response
                }
                
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def compare_texts(
        self,
        text1: str,
        text2: str,
        comparison_type: str = "semantic"
    ) -> Dict[str, Any]:
        """Compare two texts using various metrics."""
        try:
            # Get embeddings
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            # Calculate similarity
            similarity = cosine_similarity(
                [embedding1],
                [embedding2]
            )[0][0]
            
            # Analyze both texts
            analysis1 = await self.analyze_text(text1, "comparison")
            analysis2 = await self.analyze_text(text2, "comparison")
            
            # Compare entities
            entities1 = set(
                ent["text"] for ent in analysis1["analysis"]["entities"]
            )
            entities2 = set(
                ent["text"] for ent in analysis2["analysis"]["entities"]
            )
            
            common_entities = entities1.intersection(entities2)
            unique_entities1 = entities1 - entities2
            unique_entities2 = entities2 - entities1
            
            return {
                "similarity_score": float(similarity),
                "common_entities": list(common_entities),
                "unique_entities_text1": list(unique_entities1),
                "unique_entities_text2": list(unique_entities2),
                "sentiment_comparison": {
                    "text1": analysis1["analysis"]["sentiment"],
                    "text2": analysis2["analysis"]["sentiment"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text comparison failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def extract_keywords(
        self,
        text: str,
        num_keywords: int = 10
    ) -> Dict[str, Any]:
        """Extract key terms from text."""
        try:
            doc = self.nlp(text)
            
            # Extract noun phrases
            noun_phrases = [
                chunk.text.lower()
                for chunk in doc.noun_chunks
                if not all(token.is_stop for token in chunk)
            ]
            
            # Count frequencies
            phrase_freq = Counter(noun_phrases)
            
            # Get top phrases
            top_phrases = phrase_freq.most_common(num_keywords)
            
            return {
                "keywords": [
                    {
                        "phrase": phrase,
                        "frequency": freq
                    }
                    for phrase, freq in top_phrases
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def _analyze_requirements(
        self,
        text: str,
        analysis_type: str,
        requirements: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze text based on specific requirements."""
        if not self.llm_chain or not requirements:
            return {}
            
        response = await self.llm_chain.arun(
            NLPPrompt.TEXT_ANALYSIS,
            text=text[:1000],  # Limit text length
            analysis_type=analysis_type,
            requirements=json.dumps(requirements)
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse text analysis response")
            return {}
    
    def get_session_details(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about an analysis session."""
        session = self.analysis_sessions.get(session_id, {})
        return {
            "timestamp": session.get("timestamp"),
            "text_length": session.get("text_length"),
            "analysis_type": session.get("analysis_type"),
            "requirements": session.get("requirements"),
            "results": session.get("results")
        }
