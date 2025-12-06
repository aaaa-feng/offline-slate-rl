"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

环境包装器
提供统一的环境接口
"""

import torch
import pytorch_lightning as pl
import copy
from typing import Tuple, Dict

from envs.RecSim.simulators import TopicRec
from .buffer import ReplayBuffer


class EnvWrapper():
    '''
        This class provides a unified interface for gym environments, custom PyTorch environments, and model in model-based RL.
    '''
    def __init__(self, buffer: ReplayBuffer, device: torch.device, env_name: str, dyn_model: pl.LightningModule = None, **kwargs) -> None:

        self.device = device
        self.buffer = buffer
        self.obs = None
        self.done = True

        if env_name is not None:
            self.gym = False
            self.dynmod = False
            # Map env_name to environment class
            if env_name in ["topics", "TopicRec", "diffuse_topdown", "diffuse_mix", "diffuse_divpen",
                           "focused_topdown", "focused_mix", "focused_divpen"]:
                env_class = TopicRec
            else:
                raise NotImplementedError(f"Environment '{env_name}' has not been implemented.")
            self.env = env_class(device=device, **kwargs)
        elif dyn_model is not None:
            self.dynmod = True
            self.gym = False
            self.env = dyn_model
        else:
            raise ValueError("You must specify either a gym ID or a dynamics model.")

    def reset(self) -> torch.FloatTensor:
        self.done = False
        if self.dynmod:
            traj = self.buffer.sample(batch_size=1, from_data=True)
            self.obs = traj.obs[0, :]
        else:
            self.obs, info = self.env.reset()
        return self.obs

    def step(self, action: torch.Tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, Dict]:
        next_obs, reward, done, info = self.env.step(action)
        self.obs = copy.deepcopy(next_obs)

        self.done = done
        return self.obs, reward.float(), torch.tensor(done, device=self.device).long(), info

    def get_obs(self) -> Tuple[torch.FloatTensor, bool]:
        return self.obs, self.done

    def get_dimensions(self) -> Tuple[int, int]:
        return self.env.get_dimensions()

    def get_item_embeddings(self) -> torch.nn.Embedding:
        return self.env.get_item_embeddings()

    def get_random_action(self):
        return self.env.get_random_action()


def get_file_name(arg_dict):
    filename = arg_dict["agent"] + "_"
    if arg_dict["env_name"] != "Walker2DBulletEnv-v0":
        filename += arg_dict["ranker"] + "_"
        if arg_dict["env_probs"] == [0.0, 1.0, 0.0]:
            cm = "DBN_"
        else:
            cm = "MixDBN_"
        filename += cm
        if arg_dict["ranker"] in ["GeMS"]:
            ranker_checkpoint = arg_dict["ranker_checkpoint"]
            logging_policy, dataset_size, beta = ranker_checkpoint.split("_")[2:5]
            item_embedds = "_".join(ranker_checkpoint.split("_")[5:])
            filename += logging_policy + "_" + dataset_size + "_" + beta + "_" + item_embedds + "_"
        elif arg_dict["MF_checkpoint"] is not None:
            mf_checkpoint = arg_dict["MF_checkpoint"]
            mf_checkpoint = mf_checkpoint.split(".")[0]  # Remove suffix .pt
            logging_policy, dataset_size = mf_checkpoint.split("_")[1:3]
            item_embedds = "mf"
            filename += logging_policy + "_" + dataset_size + "_" + item_embedds + "_"
        else:  # True or from-scratch embeddings
            item_embedds = arg_dict["item_embedds"]
            filename += item_embedds + "_"
    else:
        filename += "walker_"
    return filename + "seed" + str(arg_dict["seed"]) + ".pt"
