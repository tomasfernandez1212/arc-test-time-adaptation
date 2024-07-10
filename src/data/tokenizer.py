from src.data.schema import *
from enum import Enum
from typing import Tuple, List
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
    START_OF_GRID = "<start-of-grid>"
    END_OF_GRID = "<end-of-grid>"
    START_OF_ROW = "<start-of-row>"
    END_OF_ROW = "<end-of-row>"

class Encoding(Enum):
    PAD = -1
    BLACK = 0
    DARK_BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    LIGHT_BLUE = 8
    BURGUNDY = 9
    START_OF_SEQUENCE = 10
    END_OF_SEQUENCE = 11
    START_OF_GRID = 12
    END_OF_GRID = 13
    START_OF_ROW = 14
    END_OF_ROW = 15

# Check if Encoding and Token Enums have the same keys
if set(Encoding.__members__.keys()) != set(Token.__members__.keys()):
    raise ValueError("Encoding and Token Enums must have the same keys.")

class TaskEncoder:
    def __init__(self, max_sequence_length: int = 9622):
        self.max_sequence_length = max_sequence_length

    def encode_task(self, task: Task) -> Tuple[List[int], List[int]]:
        encoder_sequence = self.encode_sequence(task, encoder=True)
        decoder_sequence = self.encode_sequence(task, encoder=False)
        
        # Pad sequences
        encoder_sequence = list(self.pad_sequence(encoder_sequence))
        decoder_sequence = list(self.pad_sequence(decoder_sequence))
        
        return encoder_sequence, decoder_sequence

    def pad_sequence(self, sequence: deque) -> deque:
        padding_length = self.max_sequence_length - len(sequence)
        if padding_length > 0:
            sequence.extend([Encoding.PAD.value] * padding_length)
        return sequence

    def encode_sequence(self, task: Task, encoder: bool) -> deque:
        encodings = deque([Encoding.START_OF_SEQUENCE.value])

        if encoder:
            # Tokenize All Input Grids in Train and Test
            for pair in task.train:
                encodings.extend(self.encode_grid(pair.input))
            for pair in task.test:
                encodings.extend(self.encode_grid(pair.input))
        else:
            # Tokenize All Output Grids in Train and Test
            for pair in task.train:
                encodings.extend(self.encode_grid(pair.output))
            for pair in task.test:
                encodings.extend(self.encode_grid(pair.output))

        encodings.append(Encoding.END_OF_SEQUENCE.value)
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
            encodings.append(cell)
        encodings.append(Encoding.END_OF_ROW.value)
        return encodings