"""
IbuthoAGI FastAPI Application
"""
import logging
from typing import Dict, Any
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

# Import agent components
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="IbuthoAGI API",
    description="Advanced Multi-Agent AI Problem-Solving Framework",
    version="1.0.0"
)

# Setup CORS
if config["security"]["cors_enabled"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["api"]["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Setup API Key authentication
api_key_header = APIKeyHeader(name=config["security"]["api_key_header"])

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != config.get("security", {}).get("api_key"):
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

# Initialize components
memory_manager = MemoryManager()
agent_manager = None

def initialize_agents():
    """Initialize all agents with configuration"""
    global agent_manager
    
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

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    # Initialize agents
    initialize_agents()
    
    # Setup Prometheus metrics
    if config["monitoring"]["prometheus"]["enabled"]:
        Instrumentator().instrument(app).expose(
            port=config["monitoring"]["prometheus"]["port"]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/solve", dependencies=[Depends(get_api_key)])
async def solve_problem(
    task: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Solve a complex problem using the multi-agent system
    
    Args:
        task: Dictionary containing task description and requirements
    
    Returns:
        Dictionary containing the solution and related artifacts
    """
    try:
        # Validate task input
        if "description" not in task or "requirements" not in task:
            raise HTTPException(
                status_code=400,
                detail="Task must include 'description' and 'requirements'"
            )
        
        # Process task using agent manager
        result = await agent_manager.process_task(task)
        
        return {
            "status": "success",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing task: {str(e)}"
        )

@app.get("/agents/status", dependencies=[Depends(get_api_key)])
async def get_agents_status() -> Dict[str, Any]:
    """Get status of all agents"""
    try:
        status = await agent_manager.get_status()
        return {
            "status": "success",
            "agents": status
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting agent status: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["environment"]["debug"]
    )
