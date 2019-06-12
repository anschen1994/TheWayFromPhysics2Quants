import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import torch.multiprocessing as mp 
from utils import v_wrap, push_and_pull, record, set_init
import gym 

xvaier_init = lambda x: nn.init.xavier_uniform_(x, gain=1)

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.a = nn.Linear(s_dim, 512)
        self.mu = nn.Linear(512, a_dim)
        self.sigma = nn.Linear(512, a_dim)
        self.c = nn.Linear(s_dim, 256)
        self.v = nn.Linear(256, 1)
        # xvaier_init(self.a.weight)
        # xvaier_init(self.mu.weight)
        # xvaier_init(self.sigma.weight)
        # xvaier_init(self.c.weight)
        # xvaier_init(self.v.weight)
        set_init([self.a, self.mu, self.sigma, self.c, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        y = F.relu6(self.a(x))
        mu = 2*torch.tanh(self.mu(y))
        sigma = F.softplus(self.sigma(y)) + 0.001
        c1 = F.relu6(self.c(x))
        v = self.v(c1)
        return mu, sigma, v

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        dis = self.distribution(mu.view(1, ).data, sigma.view(1,).data)
        return dis.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values  
        c_loss = td.pow(2)
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        print('log_prob size:', log_prob.size())
        print('td size:', td.size())
        print('log_prob * td size:', (log_prob * td).size())
        # entropy of Gaussian distribution
        entropy = 0.5 + 0.5 * np.log(2*np.pi) + torch.log(m.scale)
        print('entropy size:', entropy.size())
        exp_v = log_prob * td.detach() + 0.005 * entropy 
        a_loss = -exp_v 
        total_loss = (a_loss + c_loss).mean()
        return total_loss 
        

class Worker(mp.Process):
    def __init__(self, args, gnet, opt, global_ep, global_ep_r, res_queue, name, s_dim, a_dim):
        """
        args, containing MAXEPS, MAXSTEP, gamma, updateperiod
        """
        super(Worker, self).__init__()
        self.name = 'worker%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt 
        self.args = args 
        self.lnet = Net(s_dim, a_dim)
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < self.args.MAXEPS:
            # print(self.g_ep.value)
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(self.args.MAXSTEP):
                if self.name == 'worker0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None,:]))
                s_, r, done, _ = self.env.step(a.clip(-2,2))
                if t == self.args.MAXSTEP-1:
                    done = True
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8.1)/8.1)

                if total_step % self.args.updateperiod == 0 or done:
                    # print(total_step)
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.args.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        print('*')
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


                


        

