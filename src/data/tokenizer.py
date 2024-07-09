from src.data.schema import *
from src.data.dataset import ARCDataset, Split
from enum import Enum

class Tokens(Enum):
    PAD = "<padding>"
    START_OF_SEQUENCE = "<start-of-sequence>"
    END_OF_SEQUENCE = "<end-of-sequence>"
    START_OF_GRID = "<start-of-grid>"
    END_OF_GRID = "<end-of-grid>"
    START_OF_ROW = "<start-of-row>"
    END_OF_ROW = "<end-of-row>"

class Tokenizer:
    def __init__(self):
        pass 

    def tokenize_task(self, task: Task):
        encoder_sequence = self.tokenize_sequence(task, encoder=True)
        decoder_sequence = self.tokenize_sequence(task, encoder=False)
        return encoder_sequence, decoder_sequence

    def tokenize_sequence(self, task: Task, encoder: bool) -> List[str]:
        tokens = [Tokens.START_OF_SEQUENCE.value]

        if encoder:
            # Tokenize All Input Grids in Train and Test
            for pair in task.train:
                tokens.extend(self.tokenize_grid(pair.input))
            for pair in task.test:
                tokens.extend(self.tokenize_grid(pair.input))
        else:
            # Tokenize All Output Grids in Train and Test
            for pair in task.train:
                tokens.extend(self.tokenize_grid(pair.output))
            for pair in task.test:
                tokens.extend(self.tokenize_grid(pair.output))

        tokens.append(Tokens.END_OF_SEQUENCE.value)
        return tokens
            
    def tokenize_grid(self, grid: Grid) -> List[str]:
        tokens = [Tokens.START_OF_GRID.value]
        for row in grid:
            tokens.extend(self.tokenize_row(row))
        tokens.append(Tokens.END_OF_GRID.value)
        return tokens

    def tokenize_row(self, row: Row) -> List[str]:
        tokens = [Tokens.START_OF_ROW.value]
        for cell in row:
            tokens.append(str(cell))
        tokens.append(Tokens.END_OF_ROW.value)
        return tokens


