from ..main import Loop
from ..agents.agent_persona2 import AgentPersona2
from ..prompt_engine.prompt_engine import PromptEngine

class Loop2(Loop):
    def __init__(self):
        self.agent = AgentPersona2()
        self.prompt_engine = PromptEngine()

    def startLoop(self):
        print("Starting Loop 2...")
        self.brainstorming_ideas = []

    def run(self):
        self.startLoop()
        for _ in range(25):
            prompt = self.prompt_engine.generatePrompt()
            idea = self.agent.evaluate(prompt)
            self.brainstorming_ideas.append(idea)
        self.endLoop()

    def endLoop(self):
        print("Ending Loop 2...")
        print("Ideas generated: ", self.brainstorming_ideas)