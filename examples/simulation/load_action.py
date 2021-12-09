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


def parse_memory(train_dataset, action_dict, memory_replay, max_size, shuffle=True):
    loading_time = []
    last_idx = train_dataset[-1]['scene_index']

    scene_indices = list(action_dict.keys())
    if shuffle == True:
        random.shuffle(scene_indices)

    for i, scene_idx in enumerate(scene_indices):
        st = time.time()
        print('Parse scene_id {:04d} total {:04d}'.format(scene_idx, last_idx))

        rasterize_indices = train_dataset.get_scene_indices(scene_idx)
        for i, k in enumerate(rasterize_indices):
            ex = train_dataset[k]
            track_id = ex['track_id']
            frame_id = ex['frame_index']

            if track_id not in action_dict[scene_idx].keys():
                continue
            if frame_id not in action_dict[scene_idx][track_id].keys():
                continue

            action = action_dict[scene_idx][track_id][frame_id]
            state = ex['image'][image_channel, ...]
            p = i+1
            next_state = None
            while p < min(i+len(action_dict[scene_idx].keys()), len(rasterize_indices)):
                ex_next = train_dataset[rasterize_indices[p]]
                if ex_next['track_id'] == track_id and ex_next['frame_index'] == frame_id+1:
                    next_state = ex_next['image'][image_channel, ...]
                    break
                else:
                    p+=1
            if next_state is None:
                continue

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


    # # Shuffle scene idx
    # scene_indices = list(action_dict.keys())
    # random.shuffle(scene_indices)
    # for i, scene_idx in enumerate(scene_indices):
    #     st = time.time()
    #     print('Parse scene_id {:04d} among total {:04d}'.format(scene_idx, last_idx))
    #
    #     rasterize_indices = train_dataset.get_scene_indices(scene_idx)
    #     for track_id, track_info in action_dict[scene_idx].items():
    #         for frame_id, frame_info in track_info.items():
    #             for i in range(len(rasterize_indices)):
    #                 ex = train_dataset[rasterize_indices[i]]
    #                 if ex['track_id'] == track_id and ex['frame_index'] == frame_id:
    #                     frame_info['state'] = ex['image'][image_channel, ...]
    #                     rasterize_indices = np.delete(rasterize_indices, i)
    #                     break
    #
    #         frame_list = list(track_info.keys())
    #         for k in range(len(frame_list)):
    #             if k+1<len(frame_list) and frame_list[k+1] == frame_list[k]+1:
    #                 state = track_info[frame_list[k]]['state']
    #                 action = track_info[frame_list[k]]['action']
    #                 next_state = track_info[frame_list[k+1]]['state']
    #                 memory_replay.add((state, next_state, action, 1., False))
    #                 print(memory_replay.size())
    #                 if memory_replay.size() == max_size:
    #                     return memory_replay
    #
    #                 elapsed_time = time.time() - st
    #                 loading_time.append(elapsed_time)
    #                 avg_time = sum(loading_time) / len(loading_time)
    #                 print('[{}/{}] Avg time per ex: {:.3f}sec, ETA={:.2f}min'.format(memory_replay.size(), max_size,
    #                                                                                  avg_time, avg_time * (
    #                                                                                              max_size - memory_replay.size()) / 60.))
    #                 st = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config_size112.yaml', type=str)
    parser.add_argument('--channel', default=7, type=int, choices=[5, 7])
    args = parser.parse_args()
    print(args)

    os.environ["L5KIT_DATA_FOLDER"] = '.'
    data_root = '/data2/minji/dataset/l5kit_data'
    # data_root = '/data/minji/dataset/l5kit_data'
    # data_root = '/131_data/minji/l5kit_data'
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

    with open('action.pickle', 'rb') as f:
        action_dict = pickle.load(f)
    print('load')

    memory_replay = parse_memory(train_dataset, action_dict, memory_replay, REPLAY_MEMORY)
    print('total length: ', memory_replay.size())



