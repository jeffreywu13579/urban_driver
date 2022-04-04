import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import gettempdir

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer

import os
os.environ["L5KIT_DATA_FOLDER"] = "/home/jeffrey_wu13579/prediction-dataset"
dm = LocalDataManager(None)
cfg = load_config_data("./config.yaml")

# LOAD DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

vectorizer = build_vectorizer(cfg, dm)
train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer)
print(train_dataset)

train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])

# LOAD TORCHSCRIPT MODEL
URBAN_DRIVER = "Urban Driver"
OPEN_LOOP_PLANNER = "Open Loop Planner"

model_name = OPEN_LOOP_PLANNER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

if model_name == URBAN_DRIVER:
    # TODO
    print("Not implemented yet")
elif model_name == OPEN_LOOP_PLANNER:
    # with ego history
    model_path = "/home/jeffrey_wu13579/l5kit/examples/urban_driver/OL_HS.pt"
    model = torch.load(model_path)
else:
    raise ValueError(f"{model_name=} is invalid")

model = model.to(device)

# CREATE TORCH MODEL AND OPTIMIZER
weights_scaling = [1.0, 1.0, 1.0]

_num_predicted_frames = cfg["model_params"]["future_num_frames"]
_num_predicted_params = len(weights_scaling)

retrain_model = VectorizedModel(
    history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
    history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
    num_targets=_num_predicted_params * _num_predicted_frames,
    weights_scaling=weights_scaling,
    criterion=nn.L1Loss(reduction="none"),
    global_head_dropout=cfg["model_params"]["global_head_dropout"],
    disable_other_agents=cfg["model_params"]["disable_other_agents"],
    disable_map=cfg["model_params"]["disable_map"],
    disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
)
retrain_model = retrain_model.to(device)

retrain_optimizer = optim.Adam(retrain_model.parameters(), lr=1e-4)

# LOAD IN TORCHSCRIPT STATE_DICT TO OUR MODEL WITH MLP RESET
state_dict = model.state_dict()

nn.init.zeros_(state_dict['global_head.output_embed.layers.0.bias'])
nn.init.zeros_(state_dict['global_head.output_embed.layers.1.bias'])
nn.init.zeros_(state_dict['global_head.output_embed.layers.2.bias'])
nn.init.kaiming_normal_(state_dict['global_head.output_embed.layers.0.weight'], nonlinearity="relu")
nn.init.kaiming_normal_(state_dict['global_head.output_embed.layers.1.weight'], nonlinearity="relu")
nn.init.kaiming_normal_(state_dict['global_head.output_embed.layers.2.weight'], nonlinearity="relu")

retrain_model.load_state_dict(state_dict)

print(retrain_model.state_dict()['global_head.output_embed.layers.2.bias']) # sanity check

# SET ONLY MLP PARAMETERS TO REQUIRES_GRAD
retrain_model.train()
torch.set_grad_enabled(True)

for param in retrain_model.parameters():
    param.requires_grad = False

for param in retrain_model.global_head.output_embed.parameters():
    param.requires_grad = True

false_count = 0
true_count = 0
for param in retrain_model.parameters():
    if param.requires_grad:
        true_count += 1
    else:
        false_count += 1
print(false_count, true_count) # sanity check

# TRAINING LOOP
tr_it = iter(train_dataloader)
progress_bar = tqdm(range(400000))
losses_train = []

for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    # Forward pass
    data = {k: v.to(device) for k, v in data.items()}
    result = retrain_model(data)
    loss = result["loss"]
    # Backward pass
    retrain_optimizer.zero_grad()
    loss.backward()
    retrain_optimizer.step()

    if _ % 500 == 0:
        losses_train.append(loss.item())
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

# SAVE TRAINING LOSS PLOT
plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
plt.legend()
plt.savefig('/home/jeffrey_wu13579/l5kit/examples/urban_driver/retrain_loss_plot.png')


# SAVE MODEL
to_save = torch.jit.script(retrain_model.cpu())
path_to_save = "/home/jeffrey_wu13579/l5kit/examples/urban_driver/OL_HS_Retrain_10HR_2.pt"
to_save.save(path_to_save)
print(f"MODEL STORED at {path_to_save}")