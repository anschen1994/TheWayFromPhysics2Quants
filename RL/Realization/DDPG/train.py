import gym
import torch 
import torch.nn as nn 
import Noise
import Replay 
from copy import deepcopy
import numpy as np 
import matplotlib.pyplot as plt 

def train(args, actor, critic, target_actor, target_critic, actor_optm, critic_optm, env):
    #Initialize
    MSE = nn.MSELoss()
    global_step = 0
    epsilon = 1
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    memory = Replay.replayBuffer(args.buffersize)
    noise = Noise.OrnsteinUhlembeckActionNoise(mu = np.zeros(dim_action))
    plot_reward = []
    plot_policy = []
    plot_q = []
    plot_steps = []
    best_reward = -np.inf
    saved_reward = -np.inf 
    saved_ep = 0
    average_reward = 0
    saved_ep = 0
    for episode in range(args.episodes):
        s = deepcopy(env.reset())
        ep_reward = 0
        ep_q = 0
        step = 0
        for step in range(args.maxsteps):
            global_step += 1
            epsilon = max(0, epsilon - args.epsdecay)
            a = actor.select_action(torch.tensor(s, dtype=torch.float32))
            a += noise() * epsilon
            a = np.clip(a, -1, 1)
            s2, reward, done, _ = env.step(a)
            memory.add(s, a, reward, done, s2)
            if memory.num_exp > args.batchsize:
                s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(args.batchsize)
                s_batch = torch.tensor(s_batch, dtype=torch.float32)
                a_batch = torch.tensor(a_batch, dtype=torch.float32)
                r_batch = torch.tensor(r_batch, dtype=torch.float32)
                t_batch = np.array(t_batch).astype(np.float32)
                t_batch = torch.tensor(t_batch, dtype=torch.float32)
                s2_batch = torch.tensor(s2_batch, dtype=torch.float32)

                # compute loss for q value
                a2_batch = target_actor(s2_batch)
                target_q = target_critic(s2_batch, a2_batch)
                y = r_batch + args.gamma * (1.0 - t_batch) * target_q.detach()
                # print('*')
                # print('a_batch shape:', a_batch.size())
                q = critic(s_batch, a_batch)
                q_loss = MSE(q, y)
                critic_optm.zero_grad()
                q_loss.backward()
                critic_optm.step()

                # update actor
                actor_loss = -critic(s_batch, a_batch).mean()
                actor_optm.zero_grad()
                actor_loss.backward()
                actor_optm.step()

                # update target net
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1-args.tau) + args.tau * param.data)
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(target_param.data * (1-args.tau) + args.tau * param.data)
            s2 = s
            ep_reward += reward 
            if done:
                noise.reset()
                break 
        try:
            plot_reward.append([ep_reward, episode+1])
            plot_policy.append([actor_loss.data, episode+1])
            plot_q.append([q_loss.data, episode])
            plot_steps.append([step+1, episode+1])
        except:
            continue
        if ep_reward > best_reward:
            torch.save(actor.state_dict(), 'best_model_pendulum.pkl')
            best_reward = ep_reward
        print(ep_reward)
    plt.figure()
    plt.plot(np.array(plot_reward)[:,1], np.array(plot_reward)[:,0])
    plt.title('reward vs episodes')
    plt.savefig('./reward.png')
    plt.figure()
    plt.plot(np.array(plot_policy)[:,1], np.array(plot_policy)[:,0])
    plt.title('actor loss vs episodes')
    plt.savefig('./actor.png')
    plt.figure()
    plt.plot(np.array(plot_q)[:,1], np.array(plot_q)[:,0])
    plt.savefig('./q.png')
    plt.figure()
    plt.plot(np.array(plot_steps)[:,1], np.array([plot_steps])[:,0])
    plt.savefig('./steps.png')
        
        



        
    