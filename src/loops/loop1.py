```python
from ..main import Loop
from ..agents.agent_persona1 import Agent1
from ..utils.utils import generate_ideas

class Loop1(Loop):
    def __init__(self):
        self.agent = Agent1()
        super().__init__()

    def start_loop(self):
        print("Starting Loop 1...")
        self.prompt = self.prompt_engine.generate_prompt()
        self.brainstorming_ideas = []

    def run_loop(self):
        for _ in range(25):
            idea = self.agent.create(self.prompt)
            self.brainstorming_ideas.append(idea)
            self.prompt = self.prompt_engine.chain_prompts(self.prompt, idea)

    def end_loop(self):
        print("Ending Loop 1...")
        self.brainstorming_ideas = generate_ideas(self.brainstorming_ideas)
        return self.brainstorming_ideas
```