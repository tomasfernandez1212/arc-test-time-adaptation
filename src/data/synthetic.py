from src.data.schema import Task, Pair, Grid
import random
from collections import deque
import numpy as np

class SyntheticTaskGenerator:

    def __init__(self, mirrored_pairs: bool = True):
        self.mirrored_pairs = mirrored_pairs
        self.grid_dim_min = 3
        self.grid_dim_max = 30
        self.grid_dim_mean = 6
        self.num_train_pairs_min = 1
        self.num_train_pairs_max = 10
        self.num_train_pairs_mean = 3

    def _generate_grid(self, rows: int, cols: int) -> Grid:
        return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

    def _generate_pair(self, rows: int | None = None, cols: int | None = None) -> Pair:

        if rows is None or cols is None:
            rows = cols = min(max(np.random.poisson(lam=self.grid_dim_mean), self.grid_dim_min), self.grid_dim_max)

        input_grid = self._generate_grid(rows, cols)
        if self.mirrored_pairs:
            output_grid = input_grid
        else:
            # TODO - In the future, might want to develop a way to create a non-mirrored pair where the output holds a relationship to the input.
            raise NotImplementedError
        return Pair(input=input_grid, output=output_grid)
    
    def generate_task(self) -> Task:

        train_pairs = deque()
        test_pairs = deque()

        # Randomly Sample Amount of Pairs in Train
        num_train_pairs = min(max(np.random.poisson(lam=self.num_train_pairs_mean), self.num_train_pairs_min), self.num_train_pairs_max)

        for _ in range(num_train_pairs):
            train_pairs.append(self._generate_pair())
        test_pair = self._generate_pair()


        return Task(train=list(train_pairs), test=test_pair)