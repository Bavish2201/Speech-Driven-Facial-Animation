import numpy as np
import tensorflow as tf

from train import Trainer
from config import get_config
from preprocessing import get_loader, prepare_dirs_and_logger

config, unparsed = get_config()

prepare_dirs_and_logger(config)

loader = get_loader(config.data_dir, config.batch_size)
trainer = Trainer(config, loader)
trainer.train()
