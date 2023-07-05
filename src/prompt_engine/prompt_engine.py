```python
from src.prompt_engine.prompt_chain import PromptChain

class PromptEngine:
    def __init__(self):
        self.prompt_chain = PromptChain()

    def generate_prompt(self):
        return self.prompt_chain.get_next_prompt()

    def reset_prompts(self):
        self.prompt_chain.reset()

    def is_chain_empty(self):
        return self.prompt_chain.is_empty()
```