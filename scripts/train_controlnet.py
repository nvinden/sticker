import sys
import os
from copy import copy
from tqdm import tqdm
import numpy as np
import torch
import os

sys.path.append("/home/nvinden/Work/bruce/sticker")

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from configs.config_manager import set_config, print_config
from configs.config_manager import get_config_value as gcv
from sticker.data import ControlNetDataset
from packages.ControlNet.cldm.logger import ImageLogger
from packages.ControlNet.cldm.model import create_model, load_state_dict

torch.cuda.empty_cache()

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


set_config("base")

# Configs
resume_path = '/home/nvinden/Work/bruce/sticker/models/v2-pruned.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('/home/nvinden/Work/bruce/sticker/packages/ControlNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = ControlNetDataset(folds = [1, 2])
dataloader = DataLoader(dataset, num_workers=10, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], accumulate_grad_batches=4)


# Train!
trainer.fit(model, dataloader)