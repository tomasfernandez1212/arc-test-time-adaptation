from pydantic import BaseModel
from typing import List

# Define Basic Types
Cell = int
Row = List[Cell]
Grid = List[Row]

class Pair(BaseModel):
    input: Grid
    output: Grid

Pairs = List[Pair]

class Task(BaseModel):
    train: Pairs
    test: Pairs