# IbuthoAGI: Advanced Multi-Agent AI Framework

A comprehensive, modular AI system for systematic problem-solving and innovative solution generation across multiple specialized agent roles.

## Features

### Core Agents

1. **Coordinator Agent**
   - Orchestrates problem-solving workflows
   - Analyzes tasks and determines solution approaches
   - Creates and manages workflows
   - Assigns tasks to appropriate agents
   - Tracks workflow progress

2. **Researcher Agent**
   - Conducts information gathering
   - Analyzes research queries
   - Searches and synthesizes information
   - Maintains knowledge base
   - Generates research findings

3. **Planner Agent**
   - Strategic problem analysis
   - Solution design
   - Task decomposition
   - Dependency tracking
   - Critical path identification

4. **Executor Agent**
   - Task execution management
   - Resource allocation
   - Performance monitoring
   - Error handling
   - Metrics collection

5. **Critic Agent**
   - Solution evaluation
   - Performance assessment
   - Feedback generation
   - Compliance scoring
   - Trend analysis

6. **Innovator Agent**
   - Creative solution generation
   - Idea optimization
   - Innovation storage
   - Idea combination
   - Solution improvement

### Specialized Agents

7. **Code Agent**
   - Code analysis and quality assessment
   - Code generation with documentation
   - Code optimization and formatting
   - AST-based transformations
   - Semantic code search

8. **Data Science Agent**
   - Comprehensive data analysis
   - ML model training and evaluation
   - Model persistence and versioning
   - Feature engineering
   - Prediction generation

9. **NLP Agent**
   - Text analysis and entity recognition
   - Sentiment analysis
   - Text generation with style controls
   - Text comparison and similarity
   - Keyword extraction

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Quick Start

```python
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

# Initialize memory manager
memory_manager = MemoryManager()

# Initialize agents
coordinator = CoordinatorAgent("coordinator", memory_manager)
researcher = ResearcherAgent("researcher", memory_manager)
planner = PlannerAgent("planner", memory_manager)
executor = ExecutorAgent("executor", memory_manager)
critic = CriticAgent("critic", memory_manager)
innovator = InnovatorAgent("innovator", memory_manager)
code_agent = CodeAgent("code", memory_manager)
ds_agent = DataScienceAgent("data_science", memory_manager)
nlp_agent = NLPAgent("nlp", memory_manager)

# Initialize agent manager
agent_manager = AgentManager([
    coordinator, researcher, planner,
    executor, critic, innovator,
    code_agent, ds_agent, nlp_agent
])

# Start solving problems!
async def solve_problem():
    result = await coordinator.solve_task({
        "task": "Your task description here",
        "requirements": ["Your requirements here"]
    })
    return result
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Core Dependencies

- AI and Machine Learning
  - langchain>=0.0.300
  - torch>=1.9.0
  - transformers>=4.11.0
  - sentence-transformers>=2.2.0

- Vector Stores and Embeddings
  - faiss-cpu>=1.7.4
  - openai-embeddings>=0.1.0

- Data Processing
  - numpy>=1.21.0
  - pandas>=1.5.0
  - scikit-learn>=1.0.2

- NLP
  - spacy>=3.5.0
  - nltk>=3.8.1

- Code Analysis
  - black>=22.3.0
  - astor>=0.8.1

## Project Structure

```
IbuthoAGI/
├── src/
│   ├── agents/
│   │   ├── coordinator.py
│   │   ├── researcher.py
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── critic.py
│   │   ├── innovator.py
│   │   ├── code_agent.py
│   │   ├── data_science_agent.py
│   │   └── nlp_agent.py
│   ├── core/
│   ├── utils/
│   └── workflow/
├── examples/
├── tests/
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License