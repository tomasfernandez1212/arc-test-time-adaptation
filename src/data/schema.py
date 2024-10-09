from pydantic import BaseModel, conint
from typing import List

# Define Basic Types
Cell = conint(ge=0, le=9)  # Integer between 0 and 9
Row = List[Cell]
Grid = List[Row]

class Pair(BaseModel):
    input: Grid
    output: Grid

Pairs = List[Pair]

class Task(BaseModel):
    train: Pairs
    test: Pairs