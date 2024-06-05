# High Performance Language Model Dataloader

The scripts convert downloaded HuggingFace datasets into our format by tokenizing the text into context-aligned batches and compressing each one into 4 GB files with an index to look up the pieces.  This also supports a hash to check the integrity of the data.  Operations are performed in parallel with inner loops in C++ to maximize throughput.

Download and tokenize the dataset:

```bash
sudo apt install git-lfs
git lfs install

python -m dataset.download_dataset
```

Modify the copy_hosts.txt file to point to the hosts you are using for training.  Modify the top of the `copy_dataset.sh` script to change `DIR_PATH` and `SOURCE_HOST`.  All hosts for training must have a copy of the dataset.  Then copy the dataset to each host:

```bash
sudo apt install parallel rsync
cd dataset/cpp_dataloader
python setup.py install
cd ../..
```

```bash
./copy_dataset.sh
```
