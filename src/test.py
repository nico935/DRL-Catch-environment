import torch

test = [[3, 2], [1, 2], [5, 2]]
print(test)
torch_test = torch.tensor(test)
print(torch_test)
print(torch_test.shape)

# Use torch.argmax along dimension 1 (columns) to get the index of the max value for each row
lisst = torch.argmax(torch_test, dim=1).tolist()
print(lisst)
