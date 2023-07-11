import unittest
from src.main import Main

class TestMain(unittest.TestCase):

    def setUp(self):
        self.main = Main()

    def test_run(self):
        self.main.run()
        self.assertIsNotNone(self.main.brainstorming_ideas)
        self.assertTrue(len(self.main.brainstorming_ideas) > 0)

    def test_loop(self):
        self.main.run()
        self.assertIsNotNone(self.main.loop1)
        self.assertIsNotNone(self.main.loop2)
        self.assertIsNotNone(self.main.loop3)
        self.assertIsNotNone(self.main.loop4)

    def test_agents(self):
        self.main.run()
        self.assertIsNotNone(self.main.agent_persona1)
        self.assertIsNotNone(self.main.agent_persona2)
        self.assertIsNotNone(self.main.agent_persona3)
        self.assertIsNotNone(self.main.agent_persona4)
        self.assertIsNotNone(self.main.agent_persona5)
        self.assertIsNotNone(self.main.agent_persona6)
        self.assertIsNotNone(self.main.agent_persona7)
        self.assertIsNotNone(self.main.agent_persona8)

if __name__ == '__main__':
    unittest.main()