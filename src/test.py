import torch
test=[[3,2],[1,2],[5,2]]
print(test)
torch_test=torch.tensor(test)
print(torch_test)
print(torch_test.shape)
lisst=[]
for _ in torch_test:
    torch_argmax=torch.argmax(_)
    lisst.append(torch_argmax)
    print(torch_argmax)
print(lisst)
