import torch.nn as nn
import torch

emb = nn.Embedding(3,3)
print(emb)
print(emb([0,1,2]))
print(emb([0,1,2]))