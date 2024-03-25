from datasets import load_dataset

dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample")

print(dataset)
