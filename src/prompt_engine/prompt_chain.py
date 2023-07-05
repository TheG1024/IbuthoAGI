```python
class PromptChain:
    def __init__(self):
        self.chain = []

    def add_prompt(self, prompt):
        self.chain.append(prompt)

    def get_prompt(self, index):
        if index < len(self.chain):
            return self.chain[index]
        else:
            return None

    def get_chain_length(self):
        return len(self.chain)

    def get_full_chain(self):
        return self.chain
```