```python
from .agent_base import Agent

class AgentPersona8(Agent):
    def __init__(self):
        super().__init__()
        self.skills = ["Advanced Mathematics", "Physics", "Data Analysis", "Machine Learning"]
        self.background = "Data Scientist"

    def evaluate(self, problem):
        # Evaluation logic for AgentPersona8
        pass

    def create(self, problem):
        # Problem creation logic for AgentPersona8
        pass

    def solve(self, problem):
        # Problem solving logic for AgentPersona8
        pass
```