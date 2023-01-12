import torch
import torch.nn as nn
a = nn.Parameter()
print(a.device)
b = a.to(torch.device('cuda:0') ,copy=False)
print(a.device)
print(b.device)