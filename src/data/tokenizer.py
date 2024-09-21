from src.data.schema import *
from enum import Enum
from typing import List
from collections import deque

class Token(Enum):
    PAD = "<pad>"
    BLACK = "<black>"
    DARK_BLUE = "<dark-blue>"
    RED = "<red>"
    GREEN = "<green>"
    YELLOW = "<yellow>"
    GREY = "<grey>"
    MAGENTA = "<magenta>"
    ORANGE = "<orange>"
    LIGHT_BLUE = "<light-blue>"
    BURGUNDY = "<burgundy>"
    START_OF_SEQUENCE = "<start-of-sequence>"
    END_OF_SEQUENCE = "<end-of-sequence>"
    START_OF_PAIR = "<start-of-pair>"
    END_OF_PAIR = "<end-of-pair>"
    START_OF_GRID = "<start-of-grid>"
    END_OF_GRID = "<end-of-grid>"
    START_OF_ROW = "<start-of-row>"
    END_OF_ROW = "<end-of-row>"

class Encoding(Enum):
    PAD = 0
    BLACK = 1
    DARK_BLUE = 2
    RED = 3
    GREEN = 4
    YELLOW = 5
    GREY = 6
    MAGENTA = 7
    ORANGE = 8
    LIGHT_BLUE = 9
    BURGUNDY = 10
    START_OF_SEQUENCE = 11
    END_OF_SEQUENCE = 12
    START_OF_PAIR = 13
    END_OF_PAIR = 14
    START_OF_GRID = 15
    END_OF_GRID = 16
    START_OF_ROW = 17
    END_OF_ROW = 18

# Check if Encodings are positive integers
if any(encoding.value < 0 for encoding in Encoding):
    raise ValueError("Encoding values must be positive integers.")

# Check if Encoding and Token Enums have the same keys
if set(Encoding.__members__.keys()) != set(Token.__members__.keys()):
    raise ValueError("Encoding and Token Enums must have the same keys.")

class TaskEncoder:
    def __init__(self, max_sequence_length: int = 19262):
        self.max_sequence_length = max_sequence_length

    def encode_task(self, task: Task) -> List[int]:
        encoded_sequence = self.encode_sequence(task)
        
        # Pad sequence
        encoded_sequence = list(self.pad_sequence(encoded_sequence))
        
        return encoded_sequence

    def pad_sequence(self, sequence: deque) -> deque:
        padding_length = self.max_sequence_length - len(sequence)
        if padding_length > 0:
            sequence.extend([Encoding.PAD.value] * padding_length)
        return sequence

    def encode_sequence(self, task: Task) -> deque:
        encodings = deque([Encoding.START_OF_SEQUENCE.value])

        # Tokenize All Pairs in Train 
        for pair in task.train:
            encodings.extend(self.encode_pair(pair))

        # Tokenize All Pairs in Test
        for pair in task.test:
            encodings.extend(self.encode_pair(pair))

        encodings.append(Encoding.END_OF_SEQUENCE.value)
        return encodings
    
    def encode_pair(self, pair: Pair) -> deque:
        encodings = deque([Encoding.START_OF_PAIR.value])
        encodings.extend(self.encode_grid(pair.input))
        encodings.extend(self.encode_grid(pair.output))
        encodings.append(Encoding.END_OF_PAIR.value)
        return encodings
            
    def encode_grid(self, grid: Grid) -> deque:
        encodings = deque([Encoding.START_OF_GRID.value])
        for row in grid:
            encodings.extend(self.encode_row(row))
        encodings.append(Encoding.END_OF_GRID.value)
        return encodings

    def encode_row(self, row: Row) -> deque:
        encodings = deque([Encoding.START_OF_ROW.value])
        for cell in row:
            encodings.append(cell + Encoding.BLACK.value) # Add the first color's value since encodings are shifted due to other tokens
        encodings.append(Encoding.END_OF_ROW.value)
        return encodings