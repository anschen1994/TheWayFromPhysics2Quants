import argparse
import Approximator
import train 
import gym
import torch 



class NormalizeEnv(gym.ActionWrapper):
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/2
        act_b = (self.action_space.high + self.action_space.low)/2
        return act_k * action + act_b 
    
    def _reverse_action(self, action):
        act_k_inv = 2/(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/2
        return act_k_inv * (action - act_b)

parser = argparse.ArgumentParser(description='DDPG for Pendulum-v0')
parser.add_argument('--buffersize', type=int, default=100000, help='buffer size for replay memory')
parser.add_argument('--gamma', type=float, default=0.9, help='discounted factor')
parser.add_argument('--tau', type=float, default=0.001, help='parameter used to update NNs')
parser.add_argument('--lra', type=float, default=0.0001, help='learning rate for actor')
parser.add_argument('--lrc', type=float, default=0.001, help='learning rate for critic')
parser.add_argument('--h1', type=int, default=400, help='first hidden layer size')
parser.add_argument('--h2', type=int, default=300, help='second hidden layer size')
parser.add_argument('--episodes', type=int, default=10000, help='number of episodes to train')
parser.add_argument('--maxsteps', type=int, default=200, help='max steps to terminate an episode')
parser.add_argument('--epsdecay', type=float, default=1e-4, help='epsilon decay rate')
parser.add_argument('--batchsize', type=int, default=64)

args = parser.parse_args()

env = NormalizeEnv(gym.make('Pendulum-v0'))
dim_state = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]
print('State dim: {}, Action dim: {}'.format(dim_state, dim_action))

critic = Approximator.Critic(dim_state, dim_action, h1=args.h1, h2=args.h2)
actor = Approximator.Actor(dim_state, dim_action, h1=args.h1, h2=args.h2)
target_critic = Approximator.Critic(dim_state, dim_action, h1=args.h1, h2=args.h2)
target_actor = Approximator.Actor(dim_state, dim_action, h1=args.h1, h2=args.h2)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optm = torch.optim.Adam(actor.parameters(), lr=args.lra)
critic_optm = torch.optim.Adam(critic.parameters(), lr=args.lrc)

train.train(args, actor, critic, target_actor, target_critic, actor_optm, critic_optm, env)
