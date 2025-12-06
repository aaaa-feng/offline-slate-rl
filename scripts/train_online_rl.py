"""
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0
"""

import torch
import random
import pytorch_lightning as pl
from datetime import datetime

import sys
import os
from pathlib import Path
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from argparse import ArgumentParser

# 添加项目路径到sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "config"))

# 导入路径配置
from paths import (
    get_embeddings_path, get_mf_embeddings_path, get_online_dataset_path,
    get_gems_checkpoint_path, get_online_ckpt_dir, get_online_rl_results_dir
)

from common.online.data_module import BufferDataModule
from common.online.env_wrapper import EnvWrapper, get_file_name
from envs.RecSim.simulators import TopicRec
from agents.online import DQN, SAC, SlateQ, REINFORCE, REINFORCESlate, EpsGreedyOracle, RandomSlate, STOracleSlate, WolpertingerSAC
from common.online.argument_parser import MainParser
from belief_encoders.gru_belief import BeliefEncoder, GRUBelief
from rankers.gems.rankers import Ranker, TopKRanker, kHeadArgmaxRanker, GeMS
from rankers.gems.item_embeddings import ItemEmbeddings, MFEmbeddings
from training.online_loops import TrainingEpisodeLoop, ValEpisodeLoop, TestEpisodeLoop, ResettableFitLoop
from common.logger import SwanlabLogger

# Save original command line arguments for logging
_original_argv = sys.argv.copy()

######################
## Argument parsing ##
######################

main_parser = ArgumentParser()
main_parser.add_argument('--agent', type=str, required = True,
                            choices=['DQN', 'SAC', 'WolpertingerSAC', 'SlateQ', 'REINFORCE', 'REINFORCESlate',
                                        'EpsGreedyOracle', 'RandomSlate', 'STOracleSlate'], help='RL Agent.')
main_parser.add_argument('--belief', type=str, required = True,
                            choices=['none', 'GRU'], help='Belief encoder.')
main_parser.add_argument('--ranker', type=str, required = True,
                            choices=['none', 'topk', 'kargmax', 'GeMS'], help='Ranker.')
main_parser.add_argument('--item_embedds', type=str, required = True,
                            choices=['none', 'scratch', 'mf', 'ideal'], help='Item embeddings.')
main_parser.add_argument('--env_name', type=str, required = True, help='Environment.')

def get_elem(l, ch):
    for i,el in enumerate(l):
        if el.startswith(ch):
            return el
agent_name = get_elem(sys.argv, "--agent=")
belief_name = get_elem(sys.argv, "--belief=")
ranker_name = get_elem(sys.argv, "--ranker=")
embedd_name = get_elem(sys.argv, "--item_embedds=")
env_name = get_elem(sys.argv, "--env_name=")
main_args = main_parser.parse_args([agent_name, belief_name, ranker_name, embedd_name, env_name])
sys.argv.remove(agent_name)
sys.argv.remove(belief_name)
sys.argv.remove(ranker_name)
sys.argv.remove(embedd_name)

if main_args.agent == "DQN":
    agent_class = DQN
elif main_args.agent == "SAC":
    agent_class = SAC
elif main_args.agent == "WolpertingerSAC":
    agent_class = WolpertingerSAC
elif main_args.agent == "SlateQ":
    agent_class = SlateQ
elif main_args.agent == "REINFORCE":
    agent_class = REINFORCE
elif main_args.agent == "REINFORCESlate":
    agent_class = REINFORCESlate
elif main_args.agent == "EpsGreedyOracle":
    agent_class = EpsGreedyOracle
elif main_args.agent == "RandomSlate":
    agent_class = RandomSlate
elif main_args.agent == "STOracleSlate":
    agent_class = STOracleSlate
else :
    raise NotImplementedError("This agent has not been implemented yet.")

if main_args.belief in ["none"]:
    belief_class = None
elif main_args.belief == "GRU":
    belief_class = GRUBelief
else :
    raise NotImplementedError("This belief encoder has not been implemented yet.")

if main_args.ranker in ["none"]:
    ranker_class = None
elif main_args.ranker == "topk":
    ranker_class = TopKRanker
elif main_args.ranker == "kargmax":
    ranker_class = kHeadArgmaxRanker
elif main_args.ranker == "GeMS":
    ranker_class = GeMS
else :
    raise NotImplementedError("This ranker has not been implemented yet.")

if main_args.item_embedds in ["none", "ideal", "scratch"]:
    item_embedd_class = ItemEmbeddings
elif main_args.item_embedds == "mf":
    item_embedd_class = MFEmbeddings
else :
    raise NotImplementedError("This type of item embeddings has not been implemented yet.")

if main_args.env_name in ["TopicRec", "topics"]:
    env_class = TopicRec
else:
    env_class = None


argparser = MainParser() # Program-wide parameters
argparser = agent_class.add_model_specific_args(argparser)  # Agent-specific parameters
argparser = TrainingEpisodeLoop.add_model_specific_args(argparser)  # Loop-specific parameters
argparser = ValEpisodeLoop.add_model_specific_args(argparser)  # Loop-specific parameters
argparser = TestEpisodeLoop.add_model_specific_args(argparser)  # Loop-specific parameters
if belief_class is not None:
    argparser = belief_class.add_model_specific_args(argparser) # Belief-specific parameters
if env_class is not None:
    argparser = env_class.add_model_specific_args(argparser) # Env-specific parameters
if ranker_class is not None:
    argparser = ranker_class.add_model_specific_args(argparser) # Ranker-specific parameters
argparser = item_embedd_class.add_model_specific_args(argparser)  # Item embeddings-specific parameters


args = argparser.parse_args(sys.argv[1:])
arg_dict = vars(args)
arg_dict["item_embedds"] = main_args.item_embedds
logger_arg_dict = {**vars(args), **vars(main_args)}


# Print full command at the beginning
def print_full_command():
    """Print the full command that was used to run this script."""
    print("=" * 80)
    print("=== 完整命令 ===")
    print("=" * 80)
    # Reconstruct the full command
    full_cmd_parts = ["python", os.path.basename(__file__)]
    # Add all original arguments
    for arg in _original_argv[1:]:  # Skip script name
        full_cmd_parts.append(arg)
    full_cmd = " ".join(full_cmd_parts)
    print(full_cmd)
    print("=" * 80)
    print("=== 开始执行 ===")
    print("=" * 80)
    print()

# Print full command
print_full_command()

# Seeds for reproducibility
seed = int(args.seed)
pl.seed_everything(seed)

is_pomdp = (belief_class is not None)

####################
## Initialization ##
####################

# Environement and Replay Buffer
buffer = BufferDataModule(offline_data = [], **arg_dict)
env = EnvWrapper(buffer = buffer, **arg_dict)
arg_dict["env"] = env

# Item embeddings
if main_args.item_embedds in ["none"]:
    item_embeddings = None
elif main_args.item_embedds in ["scratch"]:
    item_embeddings = ItemEmbeddings.from_scratch(args.num_items, args.item_embedd_dim, device = args.device)
elif main_args.item_embedds in ["ideal"]:
    item_embeddings = ItemEmbeddings.get_from_env(env, device = args.device)
    item_embeddings.freeze()    # No fine-tuning when we already have the ideal embeddings
elif main_args.item_embedds in ["mf", "mf_fixed", "mf_init"]:
    if args.MF_checkpoint is None:
        item_embeddings = MFEmbeddings(**arg_dict)
        print("Pre-training MF embeddings ...")
        dataset_path = str(get_online_dataset_path(args.MF_dataset))
        item_embeddings.train(dataset_path, str(PROJECT_ROOT / "data"))
        arg_dict["MF_checkpoint"] = args.MF_dataset
        print("Pre-training done.")
    item_embeddings = ItemEmbeddings.from_pretrained(str(get_mf_embeddings_path(arg_dict["MF_checkpoint"])), args.device)
    if main_args.item_embedds == "mf_fixed":
        item_embeddings.freeze()
else:
    raise NotImplementedError("This type of item embeddings have not been implemented yet.")

# Belief encoder
if is_pomdp:
    if ranker_class is None:
        ranker = None
        _, action_dim, num_actions = env.get_dimensions()
    else:
        if ranker_class in [GeMS]:
            arg_dict["fixed_embedds"] = True
            if args.ranker_dataset is None :
                ranker_checkpoint = main_args.ranker + "_" + args.click_model + "_" + args.logging_policy + "_" + args.pretrain_size
            else:
                ranker_checkpoint = main_args.ranker + "_" + args.ranker_dataset
            ranker_checkpoint += "_latentdim" + str(arg_dict["latent_dim"]) + "_beta" + str(arg_dict["lambda_KL"]) + "_lambdaclick" + str(arg_dict["lambda_click"]) + \
                                    "_lambdaprior" + str(arg_dict["lambda_prior"]) + "_" + args.ranker_embedds + "_seed" + str(args.ranker_seed)
            ranker = ranker_class.load_from_checkpoint(str(get_gems_checkpoint_path(ranker_checkpoint)),
                                                    map_location = args.device, item_embeddings = item_embeddings, **arg_dict)
            ranker.freeze()
            print("Getting action bounds ...")
            if args.ranker_dataset is None :
                dataset_name = args.click_model + "_" + args.logging_policy + "_10K"
                ranker.get_action_bounds(str(get_online_dataset_path(dataset_name)))
            else:
                ranker.get_action_bounds(str(get_online_dataset_path(args.ranker_dataset)))
                            ### We find the appropriate action bounds from the aggregated posterior.
        else:
            ranker = ranker_class(item_embeddings = item_embeddings, **arg_dict)
            ranker_checkpoint = main_args.ranker
        action_dim, num_actions = ranker.get_action_dim()
    belief = belief_class(item_embeddings = ItemEmbeddings.from_scratch(args.num_items, args.item_embedd_dim, device = args.device),
                            ranker = ranker, **arg_dict)
    state_dim = belief.get_state_dim()
else:
    belief = None
    ranker = None
    state_dim, action_dim, num_actions = env.get_dimensions()

# Agent
agent = agent_class(belief = belief, ranker = ranker, state_dim = state_dim, action_dim = action_dim, num_actions = num_actions, **arg_dict)

# Print action bounds for SAC+GeMS (important for data collection)
if main_args.agent == "SAC" and ranker_class == GeMS:
    print("=" * 80)
    print("=== SAC+GeMS Action Bounds ===")
    print("=" * 80)
    if hasattr(agent, 'action_center') and hasattr(agent, 'action_scale'):
        print(f"action_center: {agent.action_center}")
        print(f"action_scale: {agent.action_scale}")
        if torch.is_tensor(agent.action_center):
            print(f"  center mean: {agent.action_center.mean().item():.4f}")
            print(f"  center std: {agent.action_center.std().item():.4f}")
        if torch.is_tensor(agent.action_scale):
            print(f"  scale mean: {agent.action_scale.mean().item():.4f}")
            print(f"  scale std: {agent.action_scale.std().item():.4f}")
    else:
        print("⚠️ Action bounds not set (will use default tanh output [-1, 1])")
    print("=" * 80)
    print()


########################
## Training procedure ##
########################

### Set SwanLab log directory
# SwanLab data will be saved to: experiments/swanlog/
if args.swan_logdir is None:
    swan_log_dir = PROJECT_ROOT / "experiments" / "swanlog"
    swan_log_dir.mkdir(parents=True, exist_ok=True)
    args.swan_logdir = str(swan_log_dir)
    print(f"📁 SwanLab directory: {args.swan_logdir}")

### Logger
logger_kwargs = {
    "project": args.swan_project or args.exp_name,
    "experiment_name": args.run_name,
    "workspace": args.swan_workspace,
    "description": args.swan_description,
    "tags": args.swan_tags,
    "config": logger_arg_dict,
    "mode": args.swan_mode,
    "logdir": args.swan_logdir,
    "run_id": args.swan_run_id,
    "resume": args.swan_resume,
}
exp_logger = SwanlabLogger(**logger_kwargs)
exp_logger.log_hyperparams(logger_arg_dict)








# ### Checkpoint
# # Use ranker_dataset for GeMS, MF_checkpoint for baselines
# checkpoint_dir_name = getattr(args, 'ranker_dataset', None) or getattr(args, 'MF_checkpoint', None) or "default"
# ckpt_dir = str(get_online_ckpt_dir(checkpoint_dir_name))
# if ranker is not None:
#     ckpt_name = args.name + "_" + ranker_checkpoint + "_agentseed" + str(seed) + "_gamma" + str(args.gamma)
#     if ranker.__class__ not in [GeMS]:
#         ckpt_name += "_rankerembedds-" + arg_dict["item_embedds"]
# else:
#     ckpt_name = args.name + "_seed" + str(seed)
#     # 只有RL算法才有gamma参数（排除Random, STOracle等简单agent）
#     if agent.__class__ not in [RandomSlate, EpsGreedyOracle, STOracleSlate] and hasattr(args, 'gamma'):
#         ckpt_name += "_gamma" + str(args.gamma)
# ckpt = ModelCheckpoint(monitor = 'val_reward', dirpath = ckpt_dir, filename = ckpt_name, mode = 'max')

# ### Agent
# trainer_agent = pl.Trainer(logger=exp_logger, enable_progress_bar = args.progress_bar, callbacks = [RichProgressBar(), ckpt],
#                             log_every_n_steps = args.log_every_n_steps, max_steps = args.max_steps + 1,
#                             check_val_every_n_epoch = args.check_val_every_n_epoch,
#                             gpus = 1 if args.device == "cuda" else None, enable_model_summary = False)


### Checkpoint Logic
# Use ranker_dataset for GeMS, MF_checkpoint for baselines
checkpoint_dir_name = getattr(args, 'ranker_dataset', None) or getattr(args, 'MF_checkpoint', None) or "default"

# 1. Determine save path
if args.save_path:
    ckpt_dir = args.save_path
    if not ckpt_dir.endswith("/"): ckpt_dir += "/"
else:
    ckpt_dir = str(get_online_ckpt_dir(checkpoint_dir_name)) + "/"
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

# 2. Determine base filename
if ranker is not None:
    base_ckpt_name = args.name + "_" + ranker_checkpoint + "_agentseed" + str(seed) + "_gamma" + str(args.gamma)
    if ranker.__class__ not in [GeMS]:
        base_ckpt_name += "_rankerembedds-" + arg_dict["item_embedds"]
else:
    base_ckpt_name = args.name + "_seed" + str(seed)
    if agent.__class__ not in [RandomSlate, EpsGreedyOracle, STOracleSlate] and hasattr(args, 'gamma'):
        base_ckpt_name += "_gamma" + str(args.gamma)

callbacks_list = [RichProgressBar()]

# 3. Callback A: Best Model (Always active)
# Suffix: _best.ckpt
ckpt_best = ModelCheckpoint(
    monitor='val_reward', 
    dirpath=ckpt_dir, 
    filename=base_ckpt_name + "_best", 
    mode='max',
    save_last=True
)
callbacks_list.append(ckpt_best)

# 4. Callback B: Step Interval (Optional)
# Suffix: _step{step}.ckpt
if args.save_every_n_steps > 0:
    ckpt_interval = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=base_ckpt_name + "_step{step}", 
        every_n_train_steps=args.save_every_n_steps,
        save_top_k=-1, # Keep ALL
        save_weights_only=False,
        save_on_train_epoch_end=False
    )
    callbacks_list.append(ckpt_interval)
    print(f"✅ Enabled interval checkpointing every {args.save_every_n_steps} steps.")

    # Keep validation frequency separate from checkpoint frequency
    # Validation will run every val_check_interval steps (default or user-specified)
    # Checkpoint will only be saved at save_every_n_steps 

### Agent
trainer_agent = pl.Trainer(
    logger=exp_logger,
    enable_progress_bar=args.progress_bar,
    callbacks=callbacks_list,
    log_every_n_steps=args.log_every_n_steps,
    max_steps=args.max_steps + 1,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    gpus=1 if args.device == "cuda" else None,
    enable_model_summary=False
)

if args.save_every_n_steps > 0:
    trainer_agent.save_step_target = args.save_every_n_steps

fit_loop = ResettableFitLoop(max_epochs_per_iter = args.iter_length_agent)
episode_loop = TrainingEpisodeLoop(env, buffer.buffer, belief, agent, ranker, random_steps = args.random_steps,
                                            max_steps = args.max_steps + 1, device = args.device)

res_dir = str(get_online_rl_results_dir(checkpoint_dir_name))
# [Fixed] Use base_ckpt_name instead of ckpt_name
val_loop = ValEpisodeLoop(belief = belief, agent = agent, ranker = ranker, trainer = trainer_agent,
                            filename_results = res_dir + "/" + base_ckpt_name + ".pt", **arg_dict)
test_loop = TestEpisodeLoop(belief = belief, agent = agent, ranker = ranker, trainer = trainer_agent,
                            filename_results = res_dir + "/" + base_ckpt_name + ".pt", **arg_dict)
trainer_agent.fit_loop.epoch_loop.val_loop.connect(val_loop)
trainer_agent.test_loop.connect(test_loop)
episode_loop.connect(batch_loop = trainer_agent.fit_loop.epoch_loop.batch_loop, val_loop = trainer_agent.fit_loop.epoch_loop.val_loop)
fit_loop.connect(episode_loop)
trainer_agent.fit_loop = fit_loop

if agent.__class__ not in [EpsGreedyOracle, RandomSlate, STOracleSlate]:
    trainer_agent.fit(agent, buffer)

    env.env.reset_random_state()
    
    # [Fixed] Load logic for final testing
    # Prioritize step model if strategy is step, otherwise best model
    if args.save_every_n_steps > 0:
        step_ckpt = ckpt_dir + base_ckpt_name + f"_step{args.save_every_n_steps}.ckpt"
        if os.path.exists(step_ckpt):
            print(f"\n### Loading specific step model for testing: {step_ckpt}")
            test_ckpt_path = step_ckpt
        else:
            print(f"⚠️ Warning: Step {args.save_every_n_steps} model not found. Falling back to best model.")
            test_ckpt_path = ckpt_dir + base_ckpt_name + "_best.ckpt"
    else:
        test_ckpt_path = ckpt_dir + base_ckpt_name + "_best.ckpt"

    print(f"### Loading model from: {test_ckpt_path}")
    
    if os.path.exists(test_ckpt_path):
        res = trainer_agent.test(model=agent, ckpt_path=test_ckpt_path, verbose=True, datamodule=buffer)
        print(f"### Test finished. Reward: {res[0]['test_reward']}")
    else:
        print(f"❌ Error: No checkpoint found to test at {test_ckpt_path}")

else:
    env.env.reset_random_state()
    res = trainer_agent.test(model=agent, verbose=True, datamodule=buffer)