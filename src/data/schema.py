from pydantic import BaseModel
from typing import List

class Grid(BaseModel):
    __root__: List[List[int]]

class Pair(BaseModel):
    input: Grid
    output: Grid

class Task(BaseModel):
    train: List[Pair]
    test: List[Pair]