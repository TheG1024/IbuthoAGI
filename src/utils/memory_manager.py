"""
Advanced memory management system for IbuthoAGI.
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

import faiss
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from redis import Redis

class MemoryManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.embeddings = OpenAIEmbeddings()
        self.conversation_memory = ConversationBufferMemory()
        
        # Initialize vector store for semantic search
        self.dimension = 1536  # OpenAI embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize Redis for distributed memory if URL provided
        self.redis = Redis.from_url(redis_url) if redis_url else None
        
        # Memory segments
        self.episodic_memory: List[Dict] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
    
    def add_to_episodic_memory(
        self,
        event: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ) -> None:
        """Add an event to episodic memory with temporal context."""
        try:
            # Generate embedding if not provided
            if embedding is None:
                event_str = json.dumps(event)
                embedding = self.embeddings.embed_query(event_str)
            
            # Add to FAISS index
            self.index.add(np.array([embedding]))
            
            # Store in episodic memory
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'embedding': embedding
            }
            self.episodic_memory.append(memory_entry)
            
            # Store in Redis if available
            if self.redis:
                key = f"episodic:{len(self.episodic_memory)-1}"
                self.redis.set(key, json.dumps(memory_entry))
            
        except Exception as e:
            print(f"Error adding to episodic memory: {str(e)}")
    
    def update_semantic_memory(
        self,
        concept: str,
        information: Dict[str, Any]
    ) -> None:
        """Update semantic memory with new information about concepts."""
        try:
            if concept not in self.semantic_memory:
                self.semantic_memory[concept] = {
                    'information': information,
                    'connections': [],
                    'last_updated': datetime.now().isoformat()
                }
            else:
                # Merge new information with existing
                self.semantic_memory[concept]['information'].update(information)
                self.semantic_memory[concept]['last_updated'] = datetime.now().isoformat()
            
            # Store in Redis if available
            if self.redis:
                key = f"semantic:{concept}"
                self.redis.set(key, json.dumps(self.semantic_memory[concept]))
            
        except Exception as e:
            print(f"Error updating semantic memory: {str(e)}")
    
    def update_working_memory(
        self,
        context_id: str,
        state: Dict[str, Any]
    ) -> None:
        """Update working memory with current context and state."""
        try:
            self.working_memory[context_id] = {
                'state': state,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in Redis if available
            if self.redis:
                key = f"working:{context_id}"
                self.redis.set(
                    key,
                    json.dumps(self.working_memory[context_id]),
                    ex=3600  # Expire after 1 hour
                )
            
        except Exception as e:
            print(f"Error updating working memory: {str(e)}")
    
    def search_episodic_memory(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search episodic memory using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search FAISS index
            D, I = self.index.search(
                np.array([query_embedding]),
                k
            )
            
            # Return matched memories
            return [
                self.episodic_memory[i]
                for i in I[0]
                if i < len(self.episodic_memory)
            ]
            
        except Exception as e:
            print(f"Error searching episodic memory: {str(e)}")
            return []
    
    def get_semantic_context(
        self,
        concepts: List[str]
    ) -> Dict[str, Any]:
        """Retrieve semantic context for given concepts."""
        try:
            context = {}
            for concept in concepts:
                if concept in self.semantic_memory:
                    context[concept] = self.semantic_memory[concept]
                    
                    # Try to get from Redis if not in local memory
                    elif self.redis:
                        redis_key = f"semantic:{concept}"
                        redis_value = self.redis.get(redis_key)
                        if redis_value:
                            context[concept] = json.loads(redis_value)
            
            return context
            
        except Exception as e:
            print(f"Error getting semantic context: {str(e)}")
            return {}
    
    def get_working_context(
        self,
        context_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve current working context."""
        try:
            # Try local memory first
            if context_id in self.working_memory:
                return self.working_memory[context_id]
            
            # Try Redis if available
            if self.redis:
                redis_key = f"working:{context_id}"
                redis_value = self.redis.get(redis_key)
                if redis_value:
                    return json.loads(redis_value)
            
            return None
            
        except Exception as e:
            print(f"Error getting working context: {str(e)}")
            return None
    
    def consolidate_memories(self) -> None:
        """Consolidate memories by finding patterns and updating semantic memory."""
        try:
            # Group similar memories
            embeddings = np.array([
                mem['embedding']
                for mem in self.episodic_memory[-100:]  # Process last 100 memories
            ])
            
            # Cluster memories
            n_clusters = min(5, len(embeddings))
            if n_clusters > 0:
                kmeans = faiss.Kmeans(
                    self.dimension,
                    n_clusters,
                    niter=20,
                    verbose=False
                )
                kmeans.train(embeddings)
                
                # Update semantic memory with patterns
                for i in range(n_clusters):
                    cluster_center = kmeans.centroids[i]
                    D, I = self.index.search(
                        np.array([cluster_center]),
                        10
                    )
                    
                    # Extract common themes
                    cluster_memories = [
                        self.episodic_memory[idx]['event']
                        for idx in I[0]
                        if idx < len(self.episodic_memory)
                    ]
                    
                    # Update semantic memory
                    self.update_semantic_memory(
                        f"pattern_{i}",
                        {
                            'memories': cluster_memories,
                            'center': cluster_center.tolist()
                        }
                    )
            
        except Exception as e:
            print(f"Error consolidating memories: {str(e)}")
    
    def prune_old_memories(self, max_age_hours: int = 24) -> None:
        """Remove old memories to prevent memory overflow."""
        try:
            current_time = datetime.now()
            self.episodic_memory = [
                mem for mem in self.episodic_memory
                if (current_time - datetime.fromisoformat(mem['timestamp'])).total_seconds() < max_age_hours * 3600
            ]
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array([
                mem['embedding']
                for mem in self.episodic_memory
            ]))
            
        except Exception as e:
            print(f"Error pruning memories: {str(e)}"))
