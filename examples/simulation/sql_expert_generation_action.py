from collections import deque
import random
import os
import numpy as np
import torch
import argparse

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.kinematic.ackerman_steering_model import fit_ackerman_model_exact, fit_ackerman_model_approximate

from numpy import linalg as LA
import math
from typing import Tuple
from scipy import optimize
from l5kit.geometry import angular_distance

import pickle
import gzip

import pdb

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def fit_ackerman_model_exact_timestep(
    x0: np.ndarray,
    y0: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gr: np.ndarray,
    gv: np.ndarray,
    wgx: np.ndarray,
    wgy: np.ndarray,
    wgr: np.ndarray,
    wgv: np.ndarray,
    timestep: float = 0.1,
    ws: float = 5.0,
    wa: float = 5.0,
    min_acc: float = -3,  # min acceleration: -3 mps2
    max_acc: float = 3,   # max acceleration: 3 mps2
    min_steer: float = -math.radians(45),  # max yaw rate: 45 degrees per second
    max_steer: float = math.radians(45),   # max yaw rate: 45 degrees per second
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits feasible ackerman-steering trajectory to groundtruth control points.
    Groundtruth is represented as 4 numpy arrays ``(gx, gy, gr, gv)``
    each of shape ``(N,)`` representing position, rotation and velocity at time i.
    Returns 4 arrays ``(x, y, r, v)`` each of shape ``(N,)`` - the optimal trajectory.
    The solution is found as minimisation of the following non-linear least squares problem:
    ::
    minimize F(steer, acc) = 0.5 * sum(
    (wgx[i] * (x[i] - gx[i])) ** 2 +
    (wgy[i] * (y[i] - gy[i])) ** 2 +
    (wgr[i] * (r[i] - gr[i])) ** 2 +
    (wgv[i] * (v[i] - gv[i])) ** 2 +
    (ws * steer[i]) ** 2 +
    (wa * acc[i]) ** 2)
    i = 1 ... N)
    subject to following unicycle motion model equations:
    x[i+1] = x[i] + cos(r[i]) * v[i]*timestep
    y[i+1] = y[i] + sin(r[i]) * v[i]*timestep
    r[i+1] = r[i] + steer[i]*timestep
    v[i+1] = v[i] + acc[i]*timestep
    min_steer < steer[i] < max_steer
    min_acc < acc[i] < max_acc
    for i = 0 .. N
    Weights ``wg*`` control adherence to the control points
    In a typical usecase ``wgx = wgy = 1`` and ``wgr = wgv = 0``
    :return: 4 arrays ``(x, y, r, v)`` each of shape ``(N,)``- the optimal trajectory.
    """
    N = len(gx)

    wsteer_acc = np.hstack([ws * np.ones(N), wa * np.ones(N)])

    def control2position(steer_acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        steer, acc = np.split(steer_acc, 2)
        r = r0 + np.cumsum(steer * timestep)
        v = v0 + np.cumsum(acc * timestep)
        x = x0 + np.cumsum(np.cos(r) * v * timestep)
        y = y0 + np.cumsum(np.sin(r) * v * timestep)
        return x, y, r, v

    def residuals(steer_acc: np.ndarray) -> np.ndarray:
        x, y, r, v = control2position(steer_acc)
        return np.hstack(
            [
                wgx * (x - gx),
                wgy * (y - gy),
                wgr * angular_distance(r, gr),
                wgv * (v - gv),
                wsteer_acc * steer_acc,
            ]
        )

    def jacobian(steer_acc: np.ndarray) -> np.ndarray:
        x, y, r, v = control2position(steer_acc)

        Jr1, Jr2, Jr3, Jr4 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        Jv1, Jv2, Jv3, Jv4 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        J = np.zeros((2 * N, 4 * N))

        for i in range(N - 1, -1, -1):
            Jr1[i:] = Jr1[i:] - np.sin(r[i]) * v[i] * timestep * wgx[i:]
            Jr2[i:] = Jr2[i:] + np.cos(r[i]) * v[i] * timestep * wgy[i:]
            Jv1[i:] = Jv1[i:] + np.cos(r[i]) * wgx[i:]
            Jv2[i:] = Jv2[i:] + np.sin(r[i]) * wgy[i:]
            Jr3[i:] = wgr[i:]
            Jv4[i:] = wgv[i:]

            J[i, :] = np.hstack([Jr1, Jr2, Jr3, Jr4])
            J[N + i, :] = np.hstack([Jv1, Jv2, Jv3, Jv4])

        return np.vstack([J.T, np.eye(N + N) * wsteer_acc])

    min_bound = np.concatenate((min_steer * np.ones(N), min_acc * np.ones(N)))
    max_bound = np.concatenate((max_steer * np.ones(N), max_acc * np.ones(N)))
    result = optimize.least_squares(residuals, np.zeros(2 * N), jacobian, (min_bound, max_bound))

    x, y, r, v = control2position(result["x"])
    steer, acc = result["x"][:N], result["x"][N:]
    return x, y, r, v, acc, steer

def fit_ackerman_model_approximate_timestep(
    gx: np.ndarray,
    gy: np.ndarray,
    gr: np.ndarray,
    gv: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    wr: np.ndarray,
    wv: np.ndarray,
    wgx: np.ndarray,
    wgy: np.ndarray,
    wgr: np.ndarray,
    wgv: np.ndarray,
    timestep: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits feasible ackerman-steering trajectory to groundtruth control points.
    Groundtruth is represented as 4 input numpy arrays ``(gx, gy, gr, gv)``
    each of size ``(N,)`` representing position, rotation and velocity at time ``i``.
    Returns 4 arrays ``(x, y, r, v)`` each of shape ``(N,)`` - the optimal trajectory.
    The solution is found as minimization of the following non-linear least squares problem:
    minimize F(x, y, r, v) = F_ground_cost(x, y, r, v) + F_kinematics_cost(x, y, r, v) where
    F_ground_cost(x, y, r, v) = 0.5 * sum(
    (wgx[i] * (x[i] - gx[i])) ** 2 +
    (wgy[i] * (y[i] - gy[i])) ** 2 +
    (wgr[i] * (r[i] - gr[i])) ** 2 +
    (wgv[i] * (v[i] - gv[i])) ** 2,
    i = 0 ... N-1)
    and
    F_kinematics_cost(x, y, r, v) = 0.5 * sum(
    (wx * (x[i] + cos(r[i]) * v[i] * timestep - x[i+1])) ** 2 +
    (wy * (y[i] + sin(r[i]) * v[i] * timestep - y[i+1])) ** 2 +
    (wr * (r[i] - r[i+1])) ** 2 +
    (wv * (v[i] - v[i+1])) ** 2,
    i = 0 ... N-2)
    Weights wg* control adherance to the control points while
    weights w* control obeying of underlying kinematic motion constrains.
    :return: 4 arrays (x, y, r, v) each of shape (N,), the optimal trajectory.
    """

    N = len(gx)

    w = np.hstack([wgx, wgy, wgr, wgv, wx, wy, wr, wv])

    def residuals(xyrv: np.ndarray) -> np.ndarray:
        x, y, r, v = np.split(xyrv, 4)

        x1, x2 = x[0:N - 1], x[1:N]
        y1, y2 = y[0:N - 1], y[1:N]
        r1, r2 = r[0:N - 1], r[1:N]
        v1, v2 = v[0:N - 1], v[1:N]

        return w * np.hstack(
            [
                x - gx,
                y - gy,
                angular_distance(r, gr),
                v * timestep - gv * timestep,
                np.append(x1 + np.cos(r1) * v1 * timestep - x2, 0),
                np.append(y1 + np.sin(r1) * v1 * timestep - y2, 0),
                np.append(angular_distance(r1, r2), 0),
                np.append(v1 * timestep - v2 * timestep, 0),
            ]
        )

    # jacobian of residuals
    def jacobian(xyrv: np.ndarray) -> np.ndarray:
        x, y, r, v = np.split(xyrv, 4)

        z = np.zeros((N, N))
        e = np.eye(N, N)
        e0 = np.block([[np.eye(N - 1, N - 1), np.zeros((N - 1, 1))], [np.zeros((1, N))]])
        e1 = np.block([[np.zeros((N - 1, 1)), np.eye(N - 1, N - 1)], [np.zeros((1, N))]])

        A = np.block(
            [
                [e, z, z, z],
                [z, e, z, z],
                [z, z, e, z],
                [z, z, z, e],
                [e0 - e1, z, -np.sin(r) * v * timestep * e0, np.cos(r) * e0],
                [z, e0 - e1, np.cos(r) * v * timestep * e0, np.sin(r) * e0],
                [z, z, e0 - e1, z],
                [z, z, z, e0 - e1],
            ]
        )

        return w[:, None] * A

    # Gauss–Newton algorithm
    xyrv = np.hstack([gx, gy, gr, gv])
    for _ in range(5):
        xyrv = xyrv - np.linalg.lstsq(jacobian(xyrv), residuals(xyrv), rcond=None)[0]
    x, y, r, v = np.split(xyrv, 4)
    return x, y, r, v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config_size112.yaml')
    parser.add_argument('--fit_model', type=str, choices=['exact', 'approx'], default='approx')
    parser.add_argument('--channel', type=int, choices=[5, 7], default=7)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--interval', type=int, default=100)
    args = parser.parse_args()

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
    # memory_replay = Memory(REPLAY_MEMORY)

    look_k_history = 6
    look_k_future = 5
    if args.channel == 5:
        image_channel = [0, 6, 12, 13, 14]
    elif args.channel == 7:
        image_channel = [0, 1, 6, 7, 12, 13, 14]
    else:
        raise NotImplementedError

    # error_pos = []
    # error_yaw = []
    # error_velocity = []
    # error_acc = []
    # error_steer = []

    print(train_dataset[-1]['scene_index'])
    last_id = train_dataset[-1]['scene_index']
    start_id = train_dataset.get_scene_indices(args.start)[0]
    end_id = train_dataset.get_scene_indices(min(args.start + args.interval - 1, last_id))[-1]
    print(start_id, end_id)

    last_id = train_dataset[-1]['scene_index']
    scene_idx_start = max(args.start, 0)
    scene_idx_end = min(args.start + args.interval - 1, last_id)

    dict = {}  # parse by scene_id -> track_id -> frame_id
    for scene_idx_ in range(scene_idx_start, scene_idx_end + 1):
        list = train_dataset.get_scene_indices(scene_idx_)
        for i in list:
            ex = train_dataset[i]
            scene_id = ex['scene_index']
            if scene_id < args.start:
                continue

            world_pose = ex['centroid'] # centroid in world reference system
            transform_matrix = ex['world_from_agent']   # mapping from agent -> world

            if scene_id not in dict.keys():
                dict[scene_id] = {}
            if ex['track_id'] not in dict[scene_id]:
                dict[scene_id][ex['track_id']] = {}

            # If frame is not sufficient, add only image (for next_state)
            availability = ex['history_availabilities'].sum() + ex['target_availabilities'].sum()
            if availability < look_k_history + look_k_future:
                continue

            # history_positions: positions in frame t, t-1, t-2, ... (in agent reference system)    =6x2
            # history_yaws: yaws in frame t, t-1, t-2, ... (in agent reference system, radian)      =6x1
            # history_velocities: v_t = {h_t - h_(t-1)}/step_time  =5x2
            # his_vel[0] = (his_pos[0]-his_pos[1])/0.1

            # target_positions: positions in future frames t+1, t+2, ..., t+50 (in agent reference system)  =50x2
            # target_yaws : yaws in future frames t+1, t+2, ..., t+50 (in agent reference system)           =50x1
            # target_availabilities: 1 (future step is valid) or 0 (future step is not valid)   =50
            # target_velocities: v_t = {t_v - t_(t-1)}/step_time   =5x2
            # tar_vel[0] = (target_pos[0]-his_pos[0])/0.1
            # target_vel[1] = (target_pos[1]-target_pos[0])/0.1

            # centroid: position in frame t (in world reference system)
            # yaw : yaw in frame t (in world reference system)

            agent_pos = np.append(ex['history_positions'][0], [1])
            world_reproduce = np.matmul(transform_matrix, agent_pos.T)[:2]

            world_pos_history = []
            for agent_pos_history in ex['history_positions'][:look_k_history]:
                extended_raster_pos_history = np.append(agent_pos_history, [1])
                world_pos_ = np.matmul(transform_matrix, extended_raster_pos_history.T)[:2]
                world_pos_history.append(world_pos_)

            world_pos_history.reverse() # t-5, t-4, ..., t

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
            yaws = yaws.reshape(-1, )   # yaws in world coordinate system (t-5, t-4, ..., t, ..., t+5)

            history_speeds = np.flip(LA.norm(ex['history_velocities'][:look_k_history], ord=2, axis=1))
            target_speeds = LA.norm(ex['target_velocities'][:look_k_future], ord=2, axis=1)
            vs = np.append(history_speeds, target_speeds)

            # crop first timestamp -> (t-4, t-3, ..., t, ..., t+5)
            xs = xs[1:]
            ys = ys[1:]
            yaws = yaws[1:]
            cur_idx = look_k_history - 2

            if args.fit_model == 'exact':
                # Input for fit_ackerman_model function should be:
                # (gx, gy, gr, gv)
                # gv[i] = norm((gx[i+1], gy[i+1]) - (gx[i], gy[i]))

                # w = np.ones_like(xs)
                # w_ = np.zeros_like(xs)
                # vs_in_timestep = vs*cfg['model_params']['step_time']
                # x, y, yaw, v, acc, steer = fit_ackerman_model_exact(xs[0], ys[0], yaws[0], vs_in_timestep[0],
                #                                                   xs, ys, yaws, vs_in_timestep,
                #                                                   w, w, w, w)
                #
                # acc = acc/cfg['model_params']['step_time']
                # curr_acc = acc[cur_idx]
                # curr_steer = steer[cur_idx]

                w = np.ones_like(xs)
                x, y, yaw, v, acc, steer = fit_ackerman_model_exact_timestep(xs[0], ys[0], yaws[0], vs[0],
                                                                  xs, ys, yaws, vs,
                                                                  w, w, w, w, ws=w, wa=w)

                curr_acc = acc[cur_idx]
                curr_steer = steer[cur_idx]

                # error
                acc_gt = vs[cur_idx + 1] - vs[cur_idx]
                steer_gt = yaws[cur_idx + 1] - yaws[cur_idx]
                # error_pos.append(np.sqrt(np.power((xs - x), 2) + np.power((ys - y), 2)).mean())
                # error_yaw.append(np.abs(yaws - yaw).mean())
                # error_velocity.append(np.abs(vs - v).mean())
                # error_acc.append(np.abs(acc_gt-curr_acc))
                # error_steer.append(np.abs(steer_gt-curr_steer))

            elif args.fit_model == 'approx':
                # w = np.ones_like(xs)
                # vs_in_timestep = vs*cfg['model_params']['step_time']
                # x, y, yaw, v = fit_ackerman_model_approximate(xs, ys, yaws, vs_in_timestep,
                #                                               w, w, w, w, w, w, w, w)
                # v = v/cfg['model_params']['step_time']
                # curr_acc = v[cur_idx+1]-v[cur_idx]
                # curr_steer = (yaw[cur_idx+1]-yaw[cur_idx])

                w = np.ones_like(xs)
                x, y, yaw, v = fit_ackerman_model_approximate_timestep(xs, ys, yaws, vs,
                                                              w, w, w, w, w, w, w, w)
                curr_acc = v[cur_idx+1]-v[cur_idx]
                curr_steer = yaw[cur_idx+1]-yaw[cur_idx]

                # Calculate error
                acc_gt = vs[cur_idx + 1] - vs[cur_idx]
                steer_gt = (yaws[cur_idx + 1] - yaws[cur_idx])
                # error_pos.append(np.sqrt(np.power((xs - x), 2) + np.power((ys - y), 2)).mean())
                # error_yaw.append(np.abs(yaws - yaw).mean())
                # error_velocity.append(np.abs(vs - v).mean())
                # error_acc.append(np.abs(acc_gt-curr_acc))
                # error_steer.append(np.abs(steer_gt-curr_steer))

            else:
                raise NotImplementedError

            print(i, ' availab:{}, track_id:{}, frame_id:{}, scene_id:{}'.format(
                availability,
                ex['track_id'],
                ex['frame_index'],
                ex['scene_index']))

            dict[scene_id][ex['track_id']][ex['frame_index']] = {'action': (curr_acc, curr_steer)}

    output_name = 'action_{}_channel{}_im{}_{:04d}to{:04d}.pickle'.format(args.fit_model, args.channel,
                                                                     cfg["raster_params"]["raster_size"][0],
                                                                     scene_idx_start, scene_idx_end)
    with open(os.path.join(data_root, output_name), 'wb') as f:
        pickle.dump(dict, f)

    # Clear dict
    dict.clear()

    # for epoch in count():
    #     # state = env.reset()
    #     # episode_reward = 0
    #     for time_steps in range(200):
    #         # env.render()
    #         action = onlineQNetwork.choose_action(state)
    #         next_state, reward, done, _ = env.step(action)
    #         memory_replay.add((state, next_state, action, 1., done))
    #         episode_reward += reward
    #         if memory_replay.size() == REPLAY_MEMORY:
    #             print('expert replay saved...')
    #             memory_replay.save('expert_replay')
    #             exit()
    #         if done:
    #             break
    #         state = next_state
    #     print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

