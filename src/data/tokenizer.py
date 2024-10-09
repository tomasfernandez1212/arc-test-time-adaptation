from src.data.schema import *
from src.data.context import MAX_TOKENS_PER_TASK
from enum import Enum
from typing import List, Tuple
from collections import deque
import torch

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
    """
    Encodes a Task into a sequence of tokens and a boolean attention matrix.
    """
    def __init__(self, max_sequence_length: int = MAX_TOKENS_PER_TASK):
        self.max_sequence_length = max_sequence_length

    def encode_task(self, task: Task) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entry point to encode a Task into a sequence of tokens and a boolean attention matrix.
        """

        encoded_sequence = deque()
        attention = torch.zeros(self.max_sequence_length, self.max_sequence_length, dtype=torch.bool)

        # Encode the sequence
        self._encode_sequence(task, encoded_sequence, attention)
        
        # Pad sequence
        self._pad_sequence(encoded_sequence)
        
        return torch.tensor(encoded_sequence, dtype=torch.long), attention

    def _pad_sequence(self, encoded_sequence: deque):
        """
        Pads the encoded sequence to the maximum sequence length.
        """
        padding_length = self.max_sequence_length - len(encoded_sequence)
        if padding_length > 0:
            encoded_sequence.extend([Encoding.PAD.value] * padding_length)

    def _encode_sequence(self, task: Task, encoded_sequence: deque, attention: torch.Tensor):
        """
        Encodes the components of a Task into a sequence of encoding values and sets the attention matrix.
        """

        # Add - Start the sequence
        encoded_sequence.append(Encoding.START_OF_SEQUENCE.value)
        start_of_sequence_index = len(encoded_sequence) - 1
        child_indices = deque()

        # Delegate - Each Training Pair 
        for pair in task.train:
            child_indices_from_pair, _ = self._encode_pair(pair, encoded_sequence, attention)
            child_indices.extend(child_indices_from_pair)

        # Delegate - The Test Pair
        child_indices_from_pair, output_grid_start_index = self._encode_pair(task.test[0], encoded_sequence, attention)
        child_indices.extend(child_indices_from_pair)

        # Add - End the sequence
        encoded_sequence.append(Encoding.END_OF_SEQUENCE.value)
        end_of_sequence_index = len(encoded_sequence) - 1

        # Set Attention - End can Attend to Start
        attention[end_of_sequence_index][start_of_sequence_index] = True 

        # Set Attention - Parent to Children & Children to Parent
        parent_indices = deque([start_of_sequence_index, end_of_sequence_index])
        i_indices, j_indices = torch.meshgrid(
            torch.tensor(parent_indices), 
            torch.tensor(child_indices), 
            indexing='ij'
        )
        attention[i_indices, j_indices] = True
        attention[j_indices, i_indices] = True

        # Set Attention - Self (Up to End of Sequence - Skip Padding)
        attention.diagonal()[:end_of_sequence_index+1] = True

        # Set Attention - Causality for Test Output Grid - Negate Non-Prefix Indices
        row_indices, col_indices = torch.triu_indices(attention.size(0), attention.size(1), offset=1) # Standard Upper Triangular Indices
        mask = col_indices >= output_grid_start_index # Only Keep Columns From Start of Output Grid
        attention[row_indices[mask], col_indices[mask]] = False # Negate These Indices

        return parent_indices


    def _encode_pair(self, pair: Pair, encoded_sequence: deque, attention: torch.Tensor) -> Tuple[deque, int]:
        """
        Given a Pair, encodes the pair into encoded_sequence and a tuple which includes a deque of the start and end indices of the pair as well as the start index of the output grid.
        """

        # Add - Start of Pair
        encoded_sequence.append(Encoding.START_OF_PAIR.value)
        start_of_pair_index = len(encoded_sequence)-1
        child_indices = deque()

        # Delegate - The Input Grid
        input_grid_indices = self._encode_grid(pair.input, encoded_sequence, attention)
        child_indices.extend(input_grid_indices)

        # Delegate - The Output Grid
        ouput_grid_indices = self._encode_grid(pair.output, encoded_sequence, attention)
        child_indices.extend(ouput_grid_indices)

        # Add - End of Pair
        encoded_sequence.append(Encoding.END_OF_PAIR.value)
        end_of_pair_index = len(encoded_sequence)-1

        # Set Attention - Start & End of Pair
        attention[start_of_pair_index][end_of_pair_index] = True 
        attention[end_of_pair_index][start_of_pair_index] = True 

        # Set Attention - Parent to Children & Children to Parent
        parent_indices = deque([start_of_pair_index, end_of_pair_index])
        i_indices, j_indices = torch.meshgrid(
            torch.tensor(parent_indices), 
            torch.tensor(child_indices), 
            indexing='ij'
        )
        attention[i_indices, j_indices] = True
        attention[j_indices, i_indices] = True

        return parent_indices, ouput_grid_indices[0]


    def _encode_grid(self, grid: Grid, encoded_sequence: deque, attention: torch.Tensor) -> deque:
        """
        Given a Grid, encodes the grid into encoded_sequence and returns a deque of the start and end indices of the grid.
        """

        # Add - Start of Grid
        encoded_sequence.append(Encoding.START_OF_GRID.value)
        start_of_grid_index = len(encoded_sequence)-1

        # Delegate - Each Row
        child_indices = deque()
        for row in grid:
            child_indices.extend(self._encode_row(row, encoded_sequence, attention))

        # Add - End of Grid
        encoded_sequence.append(Encoding.END_OF_GRID.value)
        end_of_grid_index = len(encoded_sequence)-1

        # Set Attention - Start & End of Grid 
        attention[start_of_grid_index][end_of_grid_index] = True
        attention[end_of_grid_index][start_of_grid_index] = True

        # Set Attention - Parent to Children & Children To Parent
        parent_indices = deque([start_of_grid_index, end_of_grid_index])
        i_indices, j_indices = torch.meshgrid(
            torch.tensor(parent_indices), 
            torch.tensor(child_indices), 
            indexing='ij'
        )
        attention[i_indices, j_indices] = True
        attention[j_indices, i_indices] = True

        return parent_indices

    def _encode_row(self, row: Row, encoded_sequence: deque, attention: torch.Tensor) -> deque:
        """
        Given a Row, encodes the row into encoded_sequence and returns a deque of the start and end indices of the row.
        """

        # Add - Start of Row
        encoded_sequence.append(Encoding.START_OF_ROW.value)
        start_of_row_index = len(encoded_sequence)-1

        # Delegate - Each Pixel
        child_indices = []
        for i, pixel in enumerate(row):
            encoded_sequence.append(pixel + Encoding.BLACK.value) # Add the first color's value since encodings are shifted due to other tokens
            child_indices.append(start_of_row_index+i+1)

        # Add - End of Row
        encoded_sequence.append(Encoding.END_OF_ROW.value)
        end_of_row_index = len(encoded_sequence)-1

         # Set Attention - Start & End of Row 
        attention[start_of_row_index][end_of_row_index] = True 
        attention[end_of_row_index][start_of_row_index] = True 

        # Set Attention - Parent to Children & Children to Parent
        parent_indices = deque([start_of_row_index, end_of_row_index])
        row_idx, col_idx = torch.meshgrid(
            torch.tensor(parent_indices), 
            torch.tensor(child_indices), 
            indexing='ij'
        )
        attention[row_idx, col_idx] = True
        attention[col_idx, row_idx] = True

        return parent_indices