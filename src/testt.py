import torch
x = torch.tensor([[1, 2], [3, 4]])
sum1 = x.sum(dim=1, keepdim=True)  # Output shape: (2, 1)
sum2 = x.sum(dim=1, keepdim=False) # Output shape: (2,)
print(sum1)
print(sum2)