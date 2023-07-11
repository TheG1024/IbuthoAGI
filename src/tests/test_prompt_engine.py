import unittest
from src.prompt_engine.prompt_engine import PromptEngine

class TestPromptEngine(unittest.TestCase):

    def setUp(self):
        self.prompt_engine = PromptEngine()

    def test_generate_prompt(self):
        prompt = self.prompt_engine.generate_prompt()
        self.assertIsInstance(prompt, str)

    def test_chain_prompts(self):
        prompts = self.prompt_engine.chain_prompts(25)
        self.assertIsInstance(prompts, list)
        self.assertEqual(len(prompts), 25)

    def test_prompt_engine(self):
        self.prompt_engine.run()
        self.assertTrue(self.prompt_engine.is_running)

if __name__ == '__main__':
    unittest.main()