from collections import deque
import random
import os
import numpy as np
import torch
import argparse
import time

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer

from numpy import linalg as LA
import math
from typing import Tuple
from scipy import optimize
from l5kit.geometry import angular_distance

import pickle

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


def parse_memory(train_dataset, action_dict, memory_replay, max_size, image_channel, shuffle=True):
    loading_time = []
    last_idx = train_dataset[-1]['scene_index']

    data_indices = list(action_dict.keys())
    if shuffle == True:
        random.shuffle(data_indices)

    st = time.time()
    for i, data_idx in enumerate(data_indices):
        if action_dict[data_idx]['scene_index'] != train_dataset[data_idx]['scene_index'] or \
            action_dict[data_idx]['track_id'] != train_dataset[data_idx]['track_id'] or \
            action_dict[data_idx]['frame_index'] != train_dataset[data_idx]['frame_index']:
            continue

        action = action_dict[data_idx]['action']
        state = train_dataset[data_idx]['image'][image_channel, ...]

        next_state_id = action_dict[data_idx]['next_state_id']
        next_state = train_dataset[next_state_id]['image'][image_channel, ...]

        # state, next_state, actin, reward
        memory_replay.add((state, next_state, action, 1., False))
        if memory_replay.size() == max_size:
            return memory_replay
        elapsed_time = time.time() - st
        loading_time.append(elapsed_time)
        avg_time = sum(loading_time)/len(loading_time)
        print('[{}/{}] Avg time per ex: {:.3f}sec, ETA={:.2f}min'.format(memory_replay.size(), max_size,
                                                                 avg_time, avg_time*(max_size - memory_replay.size())/60.))
        st = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config_size112.yaml', type=str)
    parser.add_argument('--channel', default=7, type=int, choices=[5, 7])
    args = parser.parse_args()
    print(args)

    os.environ["L5KIT_DATA_FOLDER"] = '.'
    data_root = '/data2/minji/dataset/l5kit_data'
    dm = LocalDataManager(data_root)
    cfg = load_config_data(args.config)
    rasterizer = build_rasterizer(cfg, dm)

    # ===== INIT DATASET
    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

    # REPLAY_MEMORY = 2500000
    REPLAY_MEMORY = 200000
    memory_replay = Memory(REPLAY_MEMORY)

    if args.channel == 5:
        image_channel = [0, 6, 12, 13, 14]
    elif args.channel == 7:
        image_channel = [0, 1, 6, 7, 12, 13, 14]
    else:
        raise NotImplementedError

    with open('action2.pickle', 'rb') as f:
        action_dict = pickle.load(f)
    print('load')

    memory_replay = parse_memory(train_dataset, action_dict, memory_replay, REPLAY_MEMORY,
                                 image_channel, shuffle=True)
    print('total length: ', memory_replay.size())