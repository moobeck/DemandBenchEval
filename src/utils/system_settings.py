
import random
import numpy as np
import torch

from src.configurations.system import SystemConfig

class SystemSettings:


    def __init__(self, config: SystemConfig):
        self.gpu = config.GPU
        self.seed = config.RANDOM_SEED



    def set_seed(self):


        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def configure_environment(self):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        print(f"Using GPU: {self.gpu}")
