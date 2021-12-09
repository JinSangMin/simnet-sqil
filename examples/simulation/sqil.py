from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from collections import deque
import random
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
import torchvision.models as models
import gym
import numpy as np
from l5kit.environment.envs.l5_env import SimulationConfigGym
from dataclasses import dataclass
import os
import pdb
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_STEER = 27
NUM_ACCEL = 9
MIN_STEER = -1
MAX_STEER = 1
MIN_ACCEL = -3
MAX_ACCEL = 3

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        np.save(path, b)

    def generate_action_space(self, ex):
        al = []
        sl = []
        for idx in range(ex.shape[0]):
            al.append(ex[idx, 2][0])
            sl.append(ex[idx, 2][1])
        accel_bin = plt.hist(al, bins=NUM_ACCEL)[1]
        accel_action_space = accel_bin[1:]*.5 + accel_bin[:-1]*.5
        steer_bin = plt.hist(sl, bins=NUM_STEER)[1]
        steer_action_space = steer_bin[1:]*.5 + steer_bin[:-1]*.5
        return accel_action_space, steer_action_space
    
    def approx_to_action_space(self, ex_action:Tuple, accel_as, steer_as):
        accel = ex_action[0]
        steer = ex_action[1]
        a_ap = accel_as[(np.abs(accel_as - accel)).argmin()]
        s_ap = steer_as[(np.abs(steer_as - steer)).argmin()]
        return (a_ap, s_ap)
    
    def load(self, path):
        ex = np.load(path+'.npy', allow_pickle=True)
        # assert(b.shape[0] == self.memory_size)
        self.a_as, self.s_as = self.generate_action_space(ex)
        
        for i in range(ex.shape[0]):
            ex[i][2] = self.approx_to_action_space(ex[i][2], self.a_as, self.s_as)
            assert ex[i][2][0] in self.a_as
            assert ex[i][2][1] in self.s_as
            self.add(ex[i])
        print('expert dataset load finished!')


class SoftQNetwork(nn.Module):
    def __init__(self):
        super(SoftQNetwork, self).__init__()
        self.alpha = 4
        resnet_version = 18
        
        if resnet_version == 18:
            self.resnet = models.resnet18(pretrained=False)
        if resnet_version == 50:
            self.resnet = models.resnet50(pretrained=False)
            
        pen_feats_num = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(7, 64, 3)
        self.resnet.fc = nn.Linear(pen_feats_num, NUM_STEER * NUM_ACCEL) # output: Q value of 27 steer bins and 9 accel bins
        
    def forward(self, x):
        return self.resnet(x)

    def getV(self, q_value):
        # 1. alpha?
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value/self.alpha), dim=1, keepdim=True))
        return v
        
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        # print('state : ', state)
        with torch.no_grad():
            try:
                q_ = self.forward(state)
                v_ = self.getV(q_).squeeze()
                # print('q & v', q, v)
                dist = torch.exp((q_-v_)/self.alpha)
                # print(dist)
                dist = dist / (torch.sum(dist))
                # print(dist)
                c = Categorical(dist)
                a = c.sample()
            except Exception as e:
                print(e)
                pdb.set_trace()
        return a.item()
    
if __name__ == "__main__":
    # env = gym.make('CartPole-v0')
    env_config_path = 'gym_config.yaml'
    rollout_sim_cfg = SimulationConfigGym()
    # rollout_sim_cfg.num_simulation_steps = None
    os.environ["L5KIT_DATA_FOLDER"] = "."
    env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                       use_kinematic=True, train=False, return_info=True, sqil=(NUM_STEER, NUM_ACCEL))
    
    onlineQNetwork = SoftQNetwork().to(device)
    targetQNetwork = SoftQNetwork().to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    GAMMA = 0.99
    REPLAY_MEMORY = 4637
    BATCH = 64
    UPDATE_STEPS = 100

    # 1. load expert replay
    expert_memory_replay = Memory(REPLAY_MEMORY//2)
    expert_memory_replay.load('expert_buffer/expert_replay_approx_channel7_im112_sample')
    accel_as = expert_memory_replay.a_as
    steer_as = expert_memory_replay.s_as
    online_memory_replay = Memory(REPLAY_MEMORY//2)
    writer = SummaryWriter('logs/sqil')

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        for time_steps in range(20):
            
            action_index_comp = onlineQNetwork.choose_action(state['image'])
            action_index = (action_index_comp // NUM_STEER, action_index_comp % NUM_STEER) # (accel, steer)
            action = np.array([accel_as[action_index[0]], steer_as[action_index[1]]])
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            online_memory_replay.add((state['image'], next_state['image'], action_index_comp, 0., done))

            if online_memory_replay.size() > 1280:
                
                # online_memory_replay.save('expert_buffer/online_sample_20')
                # pdb.set_trace()
                
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

                
                online_batch = online_memory_replay.sample(BATCH//2, False)
                online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = zip(*online_batch)

                online_batch_state = torch.FloatTensor(online_batch_state).to(device)
                online_batch_next_state = torch.FloatTensor(online_batch_next_state).to(device)
                online_batch_action = torch.FloatTensor(online_batch_action).unsqueeze(1).to(device)
                online_batch_reward = torch.FloatTensor(online_batch_reward).unsqueeze(1).to(device)
                online_batch_done = torch.FloatTensor(online_batch_done).unsqueeze(1).to(device)

                expert_batch = expert_memory_replay.sample(BATCH//2, False)
                expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = zip(*expert_batch)

                expert_batch_state = torch.FloatTensor(expert_batch_state).to(device)
                expert_batch_next_state = torch.FloatTensor(expert_batch_next_state).to(device)
                acc_idx = [np.where(accel_as == ac)[0][0] for ac in np.array(expert_batch_action)[:,0]]
                str_idx = [np.where(steer_as == st)[0][0] for st in np.array(expert_batch_action)[:,1]]
                compound_idx = [(ai*NUM_STEER + si) for ai, si  in zip(acc_idx, str_idx)]
                expert_batch_action = torch.FloatTensor(compound_idx).unsqueeze(1).to(device)
                expert_batch_reward = torch.FloatTensor(expert_batch_reward).unsqueeze(1).to(device)
                expert_batch_done = torch.FloatTensor(expert_batch_done).unsqueeze(1).to(device)

                # batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
                # batch_next_state = torch.cat([online_batch_next_state, expert_batch_next_state], dim=0)
                # batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
                # batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
                # batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)

                # pdb.set_trace()
                
                with torch.no_grad():
                    next_q = targetQNetwork(online_batch_next_state)
                    next_v = targetQNetwork.getV(next_q)
                    y = online_batch_reward + (1 - online_batch_done) * GAMMA * next_v

                # batch_action_index
                online_loss = F.mse_loss(onlineQNetwork(online_batch_state).gather(1, online_batch_action.long()), y)
                
                with torch.no_grad():
                    next_q = targetQNetwork(expert_batch_next_state)
                    next_v = targetQNetwork.getV(next_q)
                    y = expert_batch_reward + (1 - expert_batch_done) * GAMMA * next_v

                # batch_action_index
                expert_loss = F.mse_loss(onlineQNetwork(expert_batch_state).gather(1, expert_batch_action.long()), y)
                
                loss = 0.01*online_loss + expert_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), global_step=learn_steps)
            
            if done:
                break
            
            state = next_state
        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        
        if epoch % 10 == 0:
            # torch.save(onlineQNetwork.state_dict(), 'sqil-policy.para')
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
            
# online_weight = 1            
# Ep 0    Moving average score: -2437.03
# Ep 1    Moving average score: -2846.79
# Ep 2    Moving average score: -2636.63
# Ep 3    Moving average score: -2538.85
# Ep 4    Moving average score: -2040.44
# Ep 5    Moving average score: -2638.96
# Ep 6    Moving average score: -2592.03
# Ep 7    Moving average score: -2417.57
# Ep 8    Moving average score: -2654.08
# Ep 9    Moving average score: -2454.44
# Ep 10   Moving average score: -2720.15

# online_weight = 0.01
# Ep 0    Moving average score: -2643.97
# Ep 1    Moving average score: -2563.02
# Ep 2    Moving average score: -2719.71
# Ep 3    Moving average score: -2664.18
# Ep 4    Moving average score: -2795.41
# Ep 5    Moving average score: -2752.71
# Ep 6    Moving average score: -2805.65
# Ep 7    Moving average score: -2301.65
# Ep 8    Moving average score: -1965.99
# Ep 9    Moving average score: -2193.00
# Ep 10   Moving average score: -2196.87
# Ep 11   Moving average score: -1899.11
# Ep 12   Moving average score: -1751.31

                                                                                                                                                                       
# Ep 0    Moving average score: -2373.33                                                                                                                                                                                                       
# Ep 1    Moving average score: -2434.99                                                                                                                                                                                                       
# Ep 2    Moving average score: -2596.79                                                                                                                                                                                                       
# Ep 3    Moving average score: -2816.03                                                                                                                                                                                                       
# Ep 4    Moving average score: -2475.35                                                                                                                                                                                                       
# Ep 5    Moving average score: -2716.96
# Ep 6    Moving average score: -2702.79
# Ep 7    Moving average score: -2384.88
# Ep 8    Moving average score: -2605.64
# Ep 9    Moving average score: -2588.97
# Ep 10   Moving average score: -2628.37
# Ep 11   Moving average score: -2726.78
# Ep 12   Moving average score: -2521.64