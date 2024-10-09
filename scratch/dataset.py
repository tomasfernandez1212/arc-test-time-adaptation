import sys 
import os 
if os.path.basename(os.getcwd())!="arc-test-time-adaptation":
   sys.path.append('../') # Used when running directly
else:
   sys.path.append('./') # Used when running debugger

"""
This scratch script is used to test the dataset.
"""

from src.data.dataset import ARCDataset, Split

train_dataset = ARCDataset(split=Split.SYNTHETIC_MIRRORED)

encoded_sequence, attention = train_dataset[0]
print(encoded_sequence)
print(attention)
