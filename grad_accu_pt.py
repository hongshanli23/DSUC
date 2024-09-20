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
from torch.optim import AdamW
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


local_rank = 0 
# create a dataset
x = torch.randint(0, 50256, (args.train_batch_size, 128))
# repeat steps times along batch dimension
x = x.repeat(args.steps, 1)

model = create_transformer_decoder()
model = model.to(local_rank)

grad_accumulation_steps = args.train_batch_size // args.train_micro_batch_size_per_gpu
print("************** gradient_accumulation_steps:**************\n", grad_accumulation_steps)

optimizer = AdamW(model.parameters(), lr=1e-3)

log_dir = os.path.join("logs", args.run_name)
tb = SummaryWriter(log_dir)

for i in range(args.steps):
    optimizer.zero_grad()
    batch = x[i * args.train_batch_size: (i + 1) * args.train_batch_size]
    loss_per_opt_step = []
    for j in range(grad_accumulation_steps):
        inputs = batch[j * args.train_micro_batch_size_per_gpu: (j + 1) * args.train_micro_batch_size_per_gpu]
        inputs = inputs.to(local_rank)
        loss = model(inputs, labels=inputs).loss
        loss.backward()
        loss_per_opt_step.append(loss.item())
    
    loss_per_opt_step = torch.tensor(loss_per_opt_step, device=local_rank).mean()
    # divide the gradient by accumulation steps
    for param in model.parameters():
        param.grad /= grad_accumulation_steps
    
    optimizer.step()
    tb.add_scalar("loss", loss_per_opt_step.item(), i)
    
tb.close()