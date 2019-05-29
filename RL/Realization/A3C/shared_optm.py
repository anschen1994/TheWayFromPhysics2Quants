import torch 
import torch.optim as optm 
import torch.nn as nn


class SharedAdam(optm.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.9), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# see what happens in Adam
# class mymodel(nn.Module):
#     def __init__(self, in_dim=4, out_dim=2):
#         super(mymodel, self).__init__()
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.linear2 = nn.Linear(out_dim, 1)

#     def forward(self, x):
#         x = self.linear(x)
#         return self.linear2(x)

# linearmodel = mymodel(4,2)
# # print(linearmodel.parameters())
# print('state dict:', linearmodel.state_dict())
# print()
# # optimizer = optm.Adam(linearmodel.parameters())
# optimizer = SharedAdam(linearmodel.parameters())
# print('params_group:', optimizer.param_groups)
# print()
# for group in optimizer.param_groups:
#     for p in group['params']:
#         state = optimizer.state[p]
#         print('********')
#         print('p',p)
#         print('state:', state)

