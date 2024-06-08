import os
import argparse
import yaml
import time

from cpp_dataloader import DataLoader, DataVerifier

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def main(args, shard_config):
    print(f"Shard config: {shard_config}")

    if args.verify_dataset:
        print("Verifying dataset... (should take about one minute per 0.4T tokens using an SSD)")

        is_valid = DataVerifier.verify(args.dataset_dir)

        if not is_valid:
            raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    dataloader = DataLoader(args.dataset_dir, rank=args.rank, local_ranks=shard_config["rank_count"])
    if not dataloader:
        raise RuntimeError("DataLoader failed to initialize")

    dataloader.start_epoch(args.seed0, args.seed1, args.batch, args.context)

    total_microbatches = 0

    t0 = time.time()

    while True:
        batch, is_cont = dataloader.get_micro_batch()

        if batch is None:
            break

        if batch.shape[0] != args.batch:
            print(f"Batch: {batch.shape} is_cont={is_cont}")

        #print(f"Batch[0] = {batch[0][0]}")
        print(f"Batch = {batch}")

        total_microbatches += 1

        if total_microbatches % 5000 == 0:
            print(f"Reading epoch data: {total_microbatches} microbatches...")

    t1 = time.time()
    dt = t1 - t0
    print(f"Epoch complete in {dt:.2f} seconds: {total_microbatches} microbatches")

    del dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test")

    parser.add_argument("--dataset-dir", type=str, default="~/dataset_shard", help="Dataset directory")
    parser.add_argument("--verify-dataset", action="store_true", help="Verify the dataset before training")
    parser.add_argument("--rank", type=int, default=0, help="Rank to emulate")
    parser.add_argument("--batch", type=int, default=4, help="Microbatch size")
    parser.add_argument("--context", type=int, default=1024, help="Context size")
    parser.add_argument("--seed0", type=int, default=1234, help="seed0")
    parser.add_argument("--seed1", type=int, default=5678, help="seed1")

    args = parser.parse_args()

    args.dataset_dir = os.path.expanduser(args.dataset_dir)

    if not os.path.exists(args.dataset_dir):
        raise RuntimeError(f"Dataset directory {args.dataset_dir} does not exist")

    args_path = os.path.join(args.dataset_dir, "args.yml")
    shard_config = read_yaml_file(args_path)

    main(args, shard_config)
