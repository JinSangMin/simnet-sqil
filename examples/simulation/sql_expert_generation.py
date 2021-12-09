import torch
from collections import deque
import random
import numpy as np
import os
from tempfile import gettempdir
import numpy as np
import torch

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.kinematic.ackerman_steering_model import fit_ackerman_model_exact, fit_ackerman_model_approximate
from numpy import linalg as LA

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        print(b.shape)
        np.save(path, b)

    def load(self, path):
        b = np.load(path+'.npy', allow_pickle=True)
        assert(b.shape[0] == self.memory_size)

        for i in range(b.shape[0]):
            self.add(b[i])


if __name__ == '__main__':

    REPLAY_MEMORY = 25000
    memory_replay = Memory(REPLAY_MEMORY)
    
    os.environ["L5KIT_DATA_FOLDER"] = "."
    dm = LocalDataManager(None)
    cfg = load_config_data("./config.yaml")
    rasterizer = build_rasterizer(cfg, dm)

    # ===== INIT DATASET
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    pdb.set_trace()
    # Example
    look_k_history = 6
    look_k_future = 5
    for i in range(len(train_dataset)):
        ex = train_dataset[i]
        # world_pose = ex['centroid']
        transform_matrix = ex['world_from_agent']
        # assert np.array_equal(world_pose, np.matmul(transform_matrix, np.append(raster_pos, [1]).T)[:2]) # why false?

        world_pos_history = []
        for raster_pos_history in ex['history_positions'][:look_k_history]:
            extended_raster_pos_history = np.append(raster_pos_history, [1])
            world_pos_ = np.matmul(transform_matrix, extended_raster_pos_history.T)[:2]
            world_pos_history.append(world_pos_)
        
        world_pos_history.reverse()
        
        world_pos_target = []
        for target_pos in ex['target_positions'][:look_k_future]:
            extended_pos_target = np.append(target_pos, [1])
            world_pos_ = np.matmul(transform_matrix, extended_pos_target.T)[:2]
            world_pos_target.append(world_pos_)
            
        # pos shape: (look_k_history+look_k_future, 2)
        pos = np.concatenate((world_pos_history, world_pos_target)) 
        
        
        xs = np.array([xy[0] for xy in pos])
        ys = np.array([xy[1] for xy in pos])
        
        history_yaws = np.flip(ex['history_yaws'])[:look_k_history]
        target_yaws = ex['target_yaws'][:look_k_future]
        yaws = np.concatenate((history_yaws, target_yaws)) + ex['yaw'] 
        yaws = yaws.reshape(-1, )
        
        history_speeds = np.flip(LA.norm(ex['history_velocities'][:look_k_history], ord=2, axis=1))
        target_speeds = LA.norm(ex['target_velocities'][:look_k_future], ord=2, axis=1)
        vs = np.append(history_speeds, target_speeds)
        
        
        xs = xs[1:]
        ys = ys[1:]
        yaws = yaws[1:]
        #vs
        
        # why explode?
        w = np.ones_like(xs[1:])
        w_ = np.zeros_like(xs[1:])
        x, y, yaw, v, acc, steer = fit_ackerman_model_exact(xs[:-1], ys[:-1], yaws[:-1], vs[:-1],
                                                          xs[1:], ys[1:], yaws[1:], vs[1:], 
                                                          w, w, w_, w_) 
        curr_acc = acc[4]
        curr_steer = steer[4]
        
        # w = np.ones_like(xs)
        # x, y, yaw, v = fit_ackerman_model_approximate(xs, ys, yaws, vs,
        #                                               w, w, w, w, w, w, w, w)
        # curr_acc = v[5]-v[4]
        # curr_steer = yaw[5]-yaw[4]
        
        action = (curr_acc, curr_steer)
        
        state = np.take(ex['image'], (4, 10, 12, 13, 14), axis=0)
        next_state = np.take(ex['image'], (5, 11, 12, 13, 14), axis=0)
        
        
        
        done = False
        memory_replay.add((state, next_state, action, 1., done))
        
        # if memory_replay.size() == REPLAY_MEMORY:
        #     print('expert replay saved...')
        #     memory_replay.save('expert_replay')
        #     exit()
