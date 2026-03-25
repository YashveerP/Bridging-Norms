from dataclasses import dataclass

@dataclass
class model:
    """Class for keeping track of a model and its runner(local or openrouter)."""
    name: str
    runner: str

@dataclass
class prompt:
    """Class for keeping track prompts and whether they use Chain of Thought(false by default)."""
    prompt: str
    useCOT: bool = False