from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.planning.rasterized.model import RasterizedPlanningModel

import os
import glob
import pdb

os.environ["L5KIT_DATA_FOLDER"] = "."
dm = LocalDataManager(None)
cfg = load_config_data("./config.yaml")
rasterizer = build_rasterizer(cfg, dm)

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# plot some examples
for idx in range(0, len(train_dataset), len(train_dataset) // 10):
    data = train_dataset[idx]
    im = rasterizer.to_rgb(data["image"].transpose(1, 2, 0))
    target_positions = transform_points(data["target_positions"], data["raster_from_agent"])
    draw_trajectory(im, target_positions, TARGET_POINTS_COLOR)
    plt.imsave('figures/'+str(idx)+'.png', im)

globs = glob.glob('models/*')
if globs != []:
    ckpt_iters = [int(dir.split('_')[-1].split('.')[0]) for dir in globs]
    latest_iter = np.max(ckpt_iters)
    latest_iter_idx = np.argmax(ckpt_iters)
    latest_model_dir = globs[latest_iter_idx]
    model = torch.jit.load(latest_model_dir)
    print(latest_model_dir + ' has loaded!')
else:
    model = RasterizedPlanningModel(
            model_arch=cfg["model_params"]["model_architecture"],
            num_input_channels=rasterizer.num_channels(),
            num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
            weights_scaling= [1., 1., 1.],
            criterion=nn.MSELoss(reduction="none")
            )
    print(model)

train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(train_dataset)

# pdb.set_trace()
tr_it = iter(train_dataloader)
if globs != []:
    progress_bar = tqdm(range(latest_iter, cfg["train_params"]["max_num_steps"]))
else:
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))


losses_train = []
model.train()
torch.set_grad_enabled(True)

checkpoint_every_n_steps = cfg["train_params"]["checkpoint_every_n_steps"]
eval_every_n_steps =  cfg["train_params"]['eval_every_n_steps']

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}
    result = model(data)
    loss = result["loss"]
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
    
    if _ % checkpoint_every_n_steps == 0:
        to_save = torch.jit.script(model.cpu())
        path_to_save = "models/simulation_model_"+str(_)+".pt"
        to_save.save(path_to_save)
        model.cuda()
        print(f"MODEL STORED at {path_to_save}")
        
        plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
        plt.legend()
        plt.savefig('figures/loss_'+str(_)+'.png')
        plt.close()
        
    # if _ % eval_every_n_steps == 0:
        
# to_save = torch.jit.script(model.cpu())
# path_to_save = "simulation_model_.pt"
# to_save.save(path_to_save)


