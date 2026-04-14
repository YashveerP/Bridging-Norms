from dataclasses import dataclass
import pandas as pd
from typing import Callable

@dataclass
class model:
    """Class for keeping track of a model and its runner(local or openrouter)."""
    name: str
    runner: str

@dataclass
class prompt:
    """Class for keeping track prompts and the funciton to build their user prompt."""
    sysPrompt: str
    userPrompt: Callable[[str], str]
    tests: pd.DataFrame