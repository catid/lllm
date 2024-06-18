import matplotlib.pyplot as plt
import numpy as np
import math

# Define the arguments structure
class Args:
    def __init__(self, lr, steps, warmup, cooldown):
        self.lr = lr
        self.steps = steps
        self.warmup = warmup
        self.cooldown = cooldown

# Define the learning rate function
def get_lr(args, step):
    assert step <= args.steps
    # 1) linear warmup for warmup_iters steps
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    # 2) constant lr for a while
    elif step < args.steps - args.cooldown:
        return args.lr
    # 3) 1-sqrt cooldown
    else:
        decay_ratio = (step - (args.steps - args.cooldown)) / args.cooldown
        return args.lr * (1 - math.sqrt(decay_ratio))

# Initialize arguments
args = Args(lr=0.1, steps=10000, warmup=1000, cooldown=1000)

# Generate learning rate values
steps = np.arange(args.steps)
learning_rates = [get_lr(args, step) for step in steps]

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(steps, learning_rates, label='Learning Rate Schedule')
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('learning_rate_schedule.png')
plt.show()
