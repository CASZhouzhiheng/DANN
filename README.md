# DANN


## Overview
This repository contains the implementation of the DANN model using PyTorch. The model is specifically designed for predicting cancer ICI-response and is applied in the paper titled "**[Develop a deep-learning model to predict cancer immunotherapy response using in-born genomes]**" For detailed information, please refer to the paper.

## Paper Reference
If you use or refer to this DANN model in your work, please cite the following paper:
"**[Develop a deep-learning model to predict cancer immunotherapy response using in-born genomes]**"

## Requirements
PyTorch,NumPy

## Example Usage
```python
# Importing the DANN model
from DANN import DANN

# Creating an instance of the DANN model
model = DANN()

...

# Forward pass
output = model(feature)
