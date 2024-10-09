

# Define Context Window Size
MAX_PIXELS_IN_ROW = 30  # Actual is 30
MAX_PIXELS_IN_COL = 30  # Actual is 30
MAX_TRAIN_PAIRS_IN_TASK = 10  # Actual is 10

# Calculate Max Tokens per Sample
MAX_TOKENS_PER_ROW = 2 + MAX_PIXELS_IN_ROW  # Start and End of Row
MAX_TOKENS_PER_GRID = 2 + MAX_TOKENS_PER_ROW * MAX_PIXELS_IN_COL  # Start and End of Grid
MAX_TOKENS_PER_PAIR = 2 + 2 * MAX_TOKENS_PER_GRID  # Start and End of Pair, 2 Grids Per Pair
MAX_TOKENS_PER_TASK = 2 + MAX_TOKENS_PER_PAIR * (MAX_TRAIN_PAIRS_IN_TASK + 1)  # Actual is 21188

# Round Up to Be Dividible by Flex Attention's Block Size 
BLOCK_SIZE = 128
MAX_TOKENS_PER_TASK = ((MAX_TOKENS_PER_TASK + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
MAX_TOKENS_PER_TASK+=1 # Add 1 since we slice the last token
