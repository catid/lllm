from datasets import load_dataset

# default: All data
# sample-350BT: a subset randomly sampled from the whole dataset of around 350B gpt2 tokens
# sample-100BT: a subset randomly sampled from the whole dataset of around 100B gpt2 tokens
# sample-10BT: a subset randomly sampled from the whole dataset of around 10B gpt2 tokens

dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT")

# Define the total number of nodes and the current node index
total_nodes = 800  # Change this to the total number of nodes you have
current_node_index = 0  # Change this to the index of the current node (0-based)

# Shard the dataset: select only the data for the current node
sharded_dataset = dataset.shard(num_shards=total_nodes, index=current_node_index)

print(f"Sharded dataset: {len(sharded_dataset)} lines")

for i in range(len(sharded_dataset)):
    print(f"Shard {i}: {sharded_dataset[i]}")
