import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, h1, h2):
        super(Critic, self).__init__()
        self.h1 = h1
        self.linear1 = nn.Linear(dim_state, h1)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1)
        self.linear2 = nn.Linear(h1 + dim_action, h2)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1)
        self.linear3 = nn.Linear(h2, 1)
        nn.init.xavier_uniform_(self.linear3.weight, gain=1)

    def forward(self, state, action):
        x = F.elu(self.linear1(state))
        # print(x.size(), 'should be ({},{})'.format(x.size()[0], self.h1))
        x = torch.cat([x, action],1)
        # print(x.size())
        x = F.elu(self.linear2(x))
        return self.linear3(x)


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action, h1, h2):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(dim_state, h1)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(h1, h2)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear3 = nn.Linear(h2, dim_action)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x 

    def select_action(self, state):
        x = self.forward(state.unsqueeze(0))
        return x.detach().numpy()[0]

# Check 
# if __name__ =='__main__':
#     dim_state = 4
#     dim_action = 1
#     modelQ = Critic(dim_state, dim_action, 8, 4)
#     modelA = Actor(dim_state, dim_action, 10, 3)
#     state = torch.randn(10, 4)
#     action = torch.randn(10, 1)
#     q = modelQ(state, action)
#     a = modelA(state)
#     print('q shape', q.size())
#     print('a shape', a.size())
#     st = torch.tensor([1,1,1,1], dtype=torch.float32)
#     print(modelA.select_action(st))