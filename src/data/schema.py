from pydantic import BaseModel
from typing import List

class Pair(BaseModel):
    input: List
    output: List

class Task(BaseModel):
    train: List[Pair]
    test: List[Pair]