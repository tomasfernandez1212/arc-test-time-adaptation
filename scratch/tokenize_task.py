# import sys 
# import os 
# if os.path.basename(os.getcwd())!="arc-test-time-adaptation":
#    sys.path.append("../")  # Used when running directly
# else:
#    sys.path.append('./') # Used when running debugger

"""
This simple script demonstrates the tokenization of a task.
"""

from src.data.dataset import ARCDataset
from src.data.tokenizer import Tokenizer, Tokens


tokenizer = Tokenizer()
dataset = ARCDataset()

task = dataset[0]

encoder_sequence = tokenizer.tokenize_sequence(task, encoder=True)
decoder_sequence = tokenizer.tokenize_sequence(task, encoder=False)

print(encoder_sequence)
print(" ")
print(decoder_sequence)
