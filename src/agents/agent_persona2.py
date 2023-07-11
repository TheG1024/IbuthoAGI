```python
from .agent_base import Agent

class AgentPersona2(Agent):
    def __init__(self):
        super().__init__()
        self.skills = ["Problem Solving", "Creative Thinking", "Data Analysis"]
        self.background = "Data Science"

    def evaluate(self, problem):
        # Persona 2's unique evaluation method
        print(f"Agent {self.background} is evaluating the problem with {self.skills}")
        return super().evaluate(problem)

    def create(self, idea):
        # Persona 2's unique creation method
        print(f"Agent {self.background} is creating a solution with {self.skills}")
        return super().create(idea)

    def solve(self, solution):
        # Persona 2's unique solving method
        print(f"Agent {self.background} is solving the problem with {self.skills}")
        return super().solve(solution)
```