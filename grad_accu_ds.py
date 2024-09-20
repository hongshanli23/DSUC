import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel
import argparse

import json
import torch
import torch.distributed
from transformers import GPT2LMHeadModel
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from transformers import GPT2Config, GPT2LMHeadModel
import deepspeed


torch.manual_seed(0)

def create_transformer_decoder():
    config = GPT2Config(
        n_layer=4,  
        n_head=4,   
        n_embd=128  
    )
    model = GPT2LMHeadModel(config)
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--checkpoint_tag", type=str)
parser.add_argument("--universal_checkpoint", action="store_true", 
                    help="If True, load the universal checkpoint")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=8)
parser.add_argument("--train_batch_size", type=int, default=32, help="global batch size")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=100)
parser.add_argument("--run_name", type=str, default="default")
args = parser.parse_args()


# create a dataset
global_batch = torch.randint(0, 50256, (args.train_batch_size, 128))
dataset = global_batch.repeat(args.steps, 1)


model = create_transformer_decoder()

local_rank = int(os.getenv("LOCAL_RANK", -1))
assert local_rank != -1, "LOCAL_RANK environment variable must be set"

# Initialize DeepSpeed
ds_config = "deepspeed_config.json"  # Path to the JSON configuration file
with open(ds_config) as f:
    ds_config = json.load(f)

# update batch size and gradient accumulation steps
ds_config["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu 
ds_config["train_batch_size"] = args.train_batch_size
ds_config["checkpoint"]["load_universal"] = args.universal_checkpoint

world_size = int(os.getenv("WORLD_SIZE", 1))
print("************** world_size:**************\n", world_size)  
ds_config["gradient_accumulation_steps"] = args.train_batch_size // (args.train_micro_batch_size_per_gpu * world_size)
print("************** gradient_accumulation_steps:**************\n", ds_config["gradient_accumulation_steps"])

optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-3)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config=ds_config
)

log_dir = os.path.join("logs", args.run_name)
# Initialize tensorboard
assert torch.distributed.is_initialized(), "torch.distributed must be initialized"
# get global rank
global_rank = torch.distributed.get_rank()
if global_rank == 0:
    tb = SummaryWriter(log_dir)
torch.distributed.barrier()

loss_per_opt_step = []
for i in range(args.steps):
    global_batch = dataset[i * args.train_batch_size: (i + 1) * args.train_batch_size]  
    # shard the data to world_size pieces
    local_batch = global_batch[global_rank::world_size]
    local_batch = local_batch.to(local_rank)
    
    loss_per_step = []
    for j in range(ds_config["gradient_accumulation_steps"]):
        micro_local_batch = local_batch[j * args.train_micro_batch_size_per_gpu: (j + 1) * args.train_micro_batch_size_per_gpu]
        loss = model_engine(local_batch, labels=local_batch).loss
        loss_per_step.append(loss.item())
        model_engine.backward(loss)
        for param in model_engine.module.parameters():
            if param.grad is not None:
                print(param.grad)
        model_engine.step()

    loss_per_step = torch.tensor(loss_per_step, device=local_rank).mean()
    torch.distributed.all_reduce(loss_per_step, op=torch.distributed.ReduceOp.AVG)
    if global_rank == 0:
        tb.add_scalar("loss", loss_per_step.item(), i)
    

if global_rank == 0:
    tb.close()