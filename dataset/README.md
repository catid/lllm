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


## Shard and tokenize the dataset

This step will access your file server to create local shards of the dataset on each training node.  Each node will have a fraction of the dataset.

Modify the `hosts.txt` file to point to the hosts you are using for training.

```bash
cd dataset

conda activate lllm

python launch_shard_dataset.py
```

If you get an `Authentication failed` error, it is because you have not run the `playbooks/install_ssh_key.sh` script on this machine yet.
