from datasets import load_dataset

# default: All data
# sample-350BT: a subset randomly sampled from the whole dataset of around 350B gpt2 tokens
# sample-100BT: a subset randomly sampled from the whole dataset of around 100B gpt2 tokens
# sample-10BT: a subset randomly sampled from the whole dataset of around 10B gpt2 tokens

dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT")

print(dataset)
