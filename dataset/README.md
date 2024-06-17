# High Performance Tokenized Dataset Sharding/Loading

The scripts convert downloaded HuggingFace datasets into our format by tokenizing the text and storing it in a custom format.


## [Optional] Install the C++ dataloader package

The ansible scripts from `playbook/` have already installed the package.  If you need to install it manually, follow the instructions in the `dataset/package/` directory.


## Shard and tokenize the dataset

This step will access your file server to create local shards of the dataset on each training node.  Each node will have a fraction of the dataset.

Modify the `hosts.txt` file to point to the hosts you are using for training.

Update the `--dataset-dir` parameter to the location of the dataset on your file server.  The `-output-dir` will be the same on each node.

```bash
cd dataset

conda activate lllm

python make_shard_script.py --dataset-user "HuggingFaceFW" --dataset-name "fineweb-edu" --output-dir ~/dataset_shard
```

See the `make_shard_script.py --help` for more options.

This produces `run_all_hosts.sh`.  Run the dataset sharding job across the cluster:

```bash
sudo apt install pdsh parallel

./run_all_hosts.sh
```

If you hit CTRL+C it will abort the remote jobs.

This takes about ~10 hours for 4 machines on gigabit Internet using Fineweb-Edu 1.5T, and consumes 570GB disk space per node.

After this finishes, you'll have a directory called `~/dataset_shard` with the sharded dataset on each node.

Now you're ready to start training language models.  Proceed to the `train/` directory for the next steps.
