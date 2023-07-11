```python
from src.agents import agent_persona1, agent_persona2, agent_persona3, agent_persona4, agent_persona5, agent_persona6, agent_persona7, agent_persona8
from src.loops import loop1, loop2, loop3, loop4
from src.prompt_engine import prompt_engine

class Loop:
    def __init__(self, agents, prompt_engine):
        self.agents = agents
        self.prompt_engine = prompt_engine
        self.brainstorming_ideas = []

    def start(self):
        for agent in self.agents:
            self.prompt_engine.generate_prompt(agent)
            self.brainstorming_ideas.append(agent.create())

    def end(self):
        for idea in self.brainstorming_ideas:
            for agent in self.agents:
                agent.evaluate(idea)
                agent.solve(idea)

def run():
    agents = [agent_persona1.Agent(), agent_persona2.Agent(), agent_persona3.Agent(), agent_persona4.Agent(), agent_persona5.Agent(), agent_persona6.Agent(), agent_persona7.Agent(), agent_persona8.Agent()]
    prompt_engine = prompt_engine.PromptEngine()

    loop1 = Loop(agents[:2], prompt_engine)
    loop2 = Loop(agents[2:4], prompt_engine)
    loop3 = Loop(agents[4:6], prompt_engine)
    loop4 = Loop(agents[6:], prompt_engine)

    loop1.start()
    loop2.start()
    loop3.start()
    loop4.start()

    loop1.end()
    loop2.end()
    loop3.end()
    loop4.end()

if __name__ == "__main__":
    run()
```