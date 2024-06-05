# High Performance Language Model Dataloader

The scripts convert downloaded HuggingFace datasets into our format by tokenizing the text into context-aligned batches and compressing each one into 4 GB files with an index to look up the pieces.  This also supports a hash to check the integrity of the data.  Operations are performed in parallel with inner loops in C++ to maximize throughput.

## Download the dataset to your file server

Modify `download_dataset.py` script to edit the dataset location.  Download the dataset to a file server:

```bash
cd dataset

python download_dataset.py
```

If the process is aborted, run `rm -rf download_temp` to remove the temporary directory.

## Shard and tokenize the dataset

This step will access your file server to create local shards of the dataset on each training node.  Each node will have a fraction of the dataset, stored in our `cpp_dataloader` format.

Modify the `hosts.txt` file to point to the hosts you are using for training.

```bash
pip install paramiko

python install_repo.py

(1) Check out this code on all the hosts
(2) Run `python shard_dataset.py` on each host with the appropriate arguments

python shard_dataset.py

```

## 

  Modify the top of the `copy_dataset.sh` script to change `DIR_PATH` and `SOURCE_HOST`.  All hosts for training must have a copy of the dataset.  Then copy the dataset to each host: 

```bash
sudo apt install parallel rsync
cd dataset/cpp_dataloader
python setup.py install
cd ../..
```

```bash
./copy_dataset.sh
```
