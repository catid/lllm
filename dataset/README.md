# High Performance Language Model Dataloader

The scripts convert downloaded HuggingFace datasets into our format by tokenizing the text and storing it in a custom format.


## Download the dataset to your file server

Modify `download_dataset.py` script to edit the dataset location.  Download the dataset to a file server:

```bash
cd dataset

conda activate lllm

python download_dataset.py
```

If the process is aborted, run `rm -rf download_temp` to remove the temporary directory.


## [Optional] Install the C++ dataloader package

The ansible scripts from `playbook/` have already installed the package.  If you need to install it manually, follow the instructions in the `dataset/package/` directory.


## Shard and tokenize the dataset

This step will access your file server to create local shards of the dataset on each training node.  Each node will have a fraction of the dataset.

Modify the `hosts.txt` file to point to the hosts you are using for training.

Update the `--dataset-dir` parameter to the location of the dataset on your file server.  The `-output-dir` will be the same on each node.

```bash
cd dataset

conda activate lllm

python make_shard_script.py --dataset-dir /mnt/Media/datasets/fineweb-edu --output-dir ~/dataset_shard
```

Run the dataset sharding job across the cluster:

```bash
sudo apt install pdsh parallel

chmod +x run_all_hosts.sh
./run_all_hosts.sh
```
