import torch

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, filter_agents_by_frames
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI

import pdb

import os
os.environ["L5KIT_DATA_FOLDER"] = "."

# set env variable for data
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")

# ===== INIT DATASET
eval_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
mapAPI = MapAPI.from_cfg(dm ,cfg)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
print(eval_dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# simulation_model_path = "simulation_model.pt" # [0.8855176, 1.1635628, 1.7512782, 2.4170132, 3.22595, 3.819499]
# simulation_model_path = "models/simulation_model_20000.pt" # [1.653548, 2.955997, 6.050887, 9.882094, 14.720561, 20.133463]
# simulation_model_path = "models/simulation_model_280000.pt" # [1.357628, 2.1028574, 4.215295, 7.320109, 11.457499, 16.420036]
simulation_model = torch.load(simulation_model_path).to(device)
simulation_model = simulation_model.eval()

torch.set_grad_enabled(False)

scenes_to_unroll = [*range(0, 16220, 1000)] # num_scene = 16220
# scenes_to_unroll = [0, 10, 20]
num_simulation_step_example1 = 51 # np.min([len(eval_dataset.get_scene_dataset(i)) for i in range(16220)]) -> 152

sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=False, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_step_example1,
                           start_frame_index=0, show_info=True)

sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_agents=simulation_model)


sim_outs = sim_loop.unroll(scenes_to_unroll)

errors_at_step = defaultdict(list)
for sim_out in sim_outs: # for each scene
    for idx_step, agents_in_out in enumerate(sim_out.agents_ins_outs):  # for each step
        for agent_in_out in agents_in_out:  # for each agent
            annot_pos = agent_in_out.inputs["target_positions"][0]
            pred_pos = agent_in_out.outputs["positions"][0]
            if agent_in_out.inputs["target_availabilities"][0] > 0:
                errors_at_step[idx_step + 1].append(np.linalg.norm(pred_pos - annot_pos))

time_steps = np.asarray(list(errors_at_step.keys()))
errors = np.asarray([np.mean(errors_at_step[k]) for k in errors_at_step])
idxs = [4, 9, 19, 29, 39, 49]
print('scores are ', [errors[i] for i in idxs])

plt.plot(time_steps, errors, label="per step ADE")
plt.xticks(time_steps)
plt.legend()
plt.savefig('figures/ADE')
plt.close()
print('ADE figure has saved!')

