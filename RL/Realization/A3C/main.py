import argparse
from Approximator import Net, Worker
from shared_optm import SharedAdam
import torch.multiprocessing as mp 
import gym


parser = argparse.ArgumentParser('A3C for Pendulum-v0')
parser.add_argument('--gamma', type=float, default=0.9, help='discounted factor')
parser.add_argument('--MAXEPS', type=int, default=3000, help='Maximal episodes')
parser.add_argument('--MAXSTEP', type=int, default=200, help='Maximal steps')
parser.add_argument('--updateperiod', type=int, default=5, help='iteration period')

args = parser.parse_args()
env = gym.make('Pendulum-v0')
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
gnet = Net(s_dim, a_dim)
gnet.share_memory()
opt = SharedAdam(gnet.parameters(), lr=0.00002)
global_ep, global_ep_r, res_queue, = mp.Value('i', 0), mp.Value('d', 0), mp.Queue()

workers = [Worker(args, gnet, opt, global_ep, global_ep_r, res_queue, i, s_dim, a_dim) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []
while True:
    r = res_queue.get()
    if r is not None:
        res.append(r)
    else:
        break
[w.join() for w in workers]

import matplotlib.pyplot as plt 
plt.plot(res)
plt.ylabel('Moving average ep reward')
plt.xlabel('Step')
plt.show()
