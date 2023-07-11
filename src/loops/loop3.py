from ..main import Loop
from ..agents.agent_persona3 import AgentPersona3
from ..prompt_engine.prompt_engine import PromptEngine

class Loop3(Loop):
    def __init__(self, agent: AgentPersona3, prompt_engine: PromptEngine):
        super().__init__(agent, prompt_engine)

    def start_loop(self):
        print("Starting Loop 3...")
        self.brainstorming_ideas = []
        self.current_prompt = self.prompt_engine.generate_prompt()

    def run_loop(self):
        for _ in range(25):
            idea = self.agent.create(self.current_prompt)
            self.brainstorming_ideas.append(idea)
            self.current_prompt = self.prompt_engine.chain_prompts(self.current_prompt, idea)

    def end_loop(self):
        print("Ending Loop 3...")
        return self.brainstorming_ideas
