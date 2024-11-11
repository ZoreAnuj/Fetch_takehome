import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import *

# Initializing Model
model1 = SentenceTransformer('bert-base-uncased', 'mean', 512, False)
# Model with attention pooling and MLP
model2 = SentenceTransformer('bert-base-uncased', 'attention', 512, True)

# Sample Sentences
sentences = ['Anuj loves Deep Learning', 'Deep learning is a subset of Machine Learning', 'Machine Learning is a subset of Artificial Intelligence', 'Anuj loves Aritficial Intelligence']

# Lets generate embeddings!!!
embeddings1 = model1.encode(sentences)
embeddings2 = model2.encode(sentences)

print(embeddings1)
print(embeddings2)
