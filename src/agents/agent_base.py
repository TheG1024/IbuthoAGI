```python
class Agent:
    def __init__(self, skills, background):
        self.skills = skills
        self.background = background

    def evaluate(self, problem):
        raise NotImplementedError("Subclass must implement evaluate method")

    def create(self, idea):
        raise NotImplementedError("Subclass must implement create method")

    def solve(self, problem):
        raise NotImplementedError("Subclass must implement solve method")
```