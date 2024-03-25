from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import deepspeed

# Load the dataset
dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample")
print(dataset)

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=22, remove_columns=["text"])

# Define the training arguments with DeepSpeed configuration
training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    deepspeed="deepspeed_config.json",  # Path to DeepSpeed configuration file
)

# Initialize DeepSpeed
deepspeed.init_distributed()

# Create the Trainer with DeepSpeed
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model with DeepSpeed
trainer.train()
