import sys 
import os 
if os.path.basename(os.getcwd()) != "arc-test-time-adaptation":
   sys.path.append('../') # Used when running directly
   data_dir = "../../ARC-AGI/data"
else:
   sys.path.append('./') # Used when running debugger
   data_dir = "../ARC-AGI/data"

"""
This scratch script is used to test the dataset and create an interactive plot of the attention mask.
"""

from src.data.dataset import ARCDataset, Split
from src.data.tokenizer import Encoding
import numpy as np
import plotly.graph_objects as go

train_dataset = ARCDataset(split=Split.TRAIN, data_dir=data_dir)
encoded_sequence, attention_mask = train_dataset[2]

# To Numpy 
encoded_sequence = encoded_sequence.numpy()
attention_mask = attention_mask.numpy()

# Shorten
pad_value = Encoding.PAD.value
non_pad_indices = np.where(encoded_sequence != pad_value)[0]
LIMIT = non_pad_indices[-1] + 1 if non_pad_indices.size > 0 else 0
shortened_sequence = encoded_sequence[:LIMIT]
shortened_attention_mask = attention_mask[:LIMIT, :LIMIT]

# Create Interactive Hover Plot
hovertext = np.empty((LIMIT, LIMIT), dtype=object)

for i in range(LIMIT):
    for j in range(LIMIT):
        # Get the encodings at positions i and j
        query_encoding = shortened_sequence[i]
        key_encoding = shortened_sequence[j]
        # Convert encoding integers to Encoding names
        query_encoding_name = Encoding(query_encoding).name
        key_encoding_name = Encoding(key_encoding).name
        # Get the attention mask value
        mask_value = shortened_attention_mask[i, j]
        # Create hovertext
        if mask_value:
            hovertext[i, j] = (
                  f"Query: {i}, Key: {j}<br>"
                  f"Query Token: {query_encoding_name}<br>"
                  f"Key Token: {key_encoding_name}<br>"
                  f"Mask: {mask_value}"   
            )
        else:
            hovertext[i, j] = ""

# Create z-values for the heatmap
z = np.zeros((LIMIT, LIMIT), dtype=int)

for i in range(LIMIT):
    # For positions where attention_mask is True, set z to the query encoding value
    z[i, :] = np.where(shortened_attention_mask[i, :], shortened_sequence[i], 0)

# Define colors corresponding to each encoding value
colors = [
    'white',      # 0: PAD
    'black',      # 1: BLACK
    'black',   # 2: DARK_BLUE
    'black',        # 3: RED
    'black',      # 4: GREEN
    'black',     # 5: YELLOW
    'black',       # 6: GREY
    'black',    # 7: MAGENTA
    'black',     # 8: ORANGE
    'black',  # 9: LIGHT_BLUE
    'black',     #10: BURGUNDY
    'purple',     #11: START_OF_SEQUENCE
    'purple',       #12: END_OF_SEQUENCE
    'pink',      #13: START_OF_PAIR
    'pink',       #14: END_OF_PAIR
    'green',       #15: START_OF_GRID
    'green',      #16: END_OF_GRID
    'red',       #17: START_OF_ROW
    'red'        #18: END_OF_ROW
]

# Create a colorscale mapping encoding values to colors
colorscale = []

for i in range(len(colors)):
    fraction = i / (len(colors) - 1)
    colorscale.append([fraction, colors[i]])

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=z,
    x=np.arange(LIMIT),
    y=np.arange(LIMIT),
    text=hovertext,
    hoverinfo='text',
    colorscale=colorscale,
    zmin=0,
    zmax=18,
    showscale=False
))

fig.update_layout(
    xaxis_nticks=10,
    yaxis_nticks=10,
    yaxis=dict(autorange='reversed'),  # Reverse the y-axis
    width=800,  
    height=800, 
    title="Attention Mask",
    title_font=dict(size=24),  
    xaxis_title="Key",  
    yaxis_title="Query",    
    xaxis_title_font=dict(size=18),  
    yaxis_title_font=dict(size=18),
    hoverlabel=dict(font_size=16)  # Increase hover text font size
)

fig.show()

