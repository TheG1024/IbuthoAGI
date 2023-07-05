from ..main import Loop
from ..agents.agent_persona4 import AgentPersona4
from ..prompt_engine.prompt_engine import PromptEngine

class Loop4(Loop):
    def __init__(self, agent: AgentPersona4, prompt_engine: PromptEngine):
        super().__init__(agent, prompt_engine)

    def start_loop(self):
        print("Starting Loop 4...")
        self.brainstorming_ideas = []
        self.prompt = self.prompt_engine.generate_prompt()

        for _ in range(25):
            idea = self.agent.create(self.prompt)
            self.brainstorming_ideas.append(idea)
            self.prompt = self.prompt_engine.chain_prompt(self.prompt, idea)

    def end_loop(self):
        print("Ending Loop 4...")
        self.agent.evaluate(self.brainstorming_ideas)
        self.agent.solve(self.brainstorming_ideas)