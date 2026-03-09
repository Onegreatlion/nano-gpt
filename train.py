import os
import math
import numpy as np
import torch
from model import GPTConfig, GPT

# --- 1. Hyperparameters ---
batch_size = 4 
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 6e-4

# Optimization specific parameters
grad_accum_steps = 4 
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 6e-5 # 10% of learning_rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Data Loader ---
data_dir = os.path.dirname(__file__)

def get_batch(split):
    # We load the binary files we created in data.py
    # np.memmap reads directly from the hard drive without filling up RAM
    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Generate random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract chunks of data
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y

# --- 3. Initialize Model ---
config = GPTConfig()
model = GPT(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

# --- 4. Learning Rate Scheduler (The Math) ---
def get_lr(it):
    # 1. Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2. Minimum learning rate after decay
    if it > lr_decay_iters:
        return min_lr
    # 3. Cosine decay wave
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# --- 5. The Training Loop ---
X, Y = get_batch('train') # Fetch very first batch

print(f'Starting training on {device}!')

for step in range(max_iters):
    
    # Set the learning rate for this specific step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Gradient Accumulation: We run multiple micro-batches before taking a single optimization step
    for micro_step in range(grad_accum_steps):
        # Mixed Precision: Do the heavy math in 16-bit for speed
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16):
            logits, loss = model(X, Y)
            # Scale the loss down because we are accumulating gradients
            loss = loss / grad_accum_steps 
        
        loss.backward()
        X, Y = get_batch('train') # Prefetch next batch
        
    # Clip gradients to prevent "exploding" math
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 10 == 0:
        print(f'Step {step} | Loss: {loss.item() * grad_accum_steps:.4f} | LR: {lr:.6f}')
