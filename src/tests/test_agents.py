import unittest
from src.agents.agent_base import Agent
from src.agents.agent_persona1 import AgentPersona1
from src.agents.agent_persona2 import AgentPersona2
from src.agents.agent_persona3 import AgentPersona3
from src.agents.agent_persona4 import AgentPersona4
from src.agents.agent_persona5 import AgentPersona5
from src.agents.agent_persona6 import AgentPersona6
from src.agents.agent_persona7 import AgentPersona7
from src.agents.agent_persona8 import AgentPersona8

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.agent1 = AgentPersona1()
        self.agent2 = AgentPersona2()
        self.agent3 = AgentPersona3()
        self.agent4 = AgentPersona4()
        self.agent5 = AgentPersona5()
        self.agent6 = AgentPersona6()
        self.agent7 = AgentPersona7()
        self.agent8 = AgentPersona8()

    def test_evaluate(self):
        self.assertIsNotNone(self.agent1.evaluate())
        self.assertIsNotNone(self.agent2.evaluate())
        self.assertIsNotNone(self.agent3.evaluate())
        self.assertIsNotNone(self.agent4.evaluate())
        self.assertIsNotNone(self.agent5.evaluate())
        self.assertIsNotNone(self.agent6.evaluate())
        self.assertIsNotNone(self.agent7.evaluate())
        self.assertIsNotNone(self.agent8.evaluate())

    def test_create(self):
        self.assertIsNotNone(self.agent1.create())
        self.assertIsNotNone(self.agent2.create())
        self.assertIsNotNone(self.agent3.create())
        self.assertIsNotNone(self.agent4.create())
        self.assertIsNotNone(self.agent5.create())
        self.assertIsNotNone(self.agent6.create())
        self.assertIsNotNone(self.agent7.create())
        self.assertIsNotNone(self.agent8.create())

    def test_solve(self):
        self.assertIsNotNone(self.agent1.solve())
        self.assertIsNotNone(self.agent2.solve())
        self.assertIsNotNone(self.agent3.solve())
        self.assertIsNotNone(self.agent4.solve())
        self.assertIsNotNone(self.agent5.solve())
        self.assertIsNotNone(self.agent6.solve())
        self.assertIsNotNone(self.agent7.solve())
        self.assertIsNotNone(self.agent8.solve())

if __name__ == '__main__':
    unittest.main()