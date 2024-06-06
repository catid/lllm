import argparse
import os

# Function to read hosts from the file
def read_hosts(file_path):
    hosts = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                hostname, rank_count = line.split()
                hosts.append((hostname, int(rank_count)))
    return hosts

# Function to generate the master shell script using pdsh
def generate_master_script(hosts, world_size, args):
    script_content = "#!/bin/bash\n"
    script_content += "# Master script to execute shard_dataset.py on multiple hosts using pdsh\n\n"

    rank_start = 0
    for hostname, rank_count in hosts:
        command = (
            f"~/mambaforge/envs/{args.conda_env}/bin/python ~/lllm/dataset/shard_dataset.py "
            f"--dataset_dir {args.dataset_dir} "
            f"--rank_start {rank_start} "
            f"--rank_count {rank_count} "
            f"--world_size {world_size} "
            f"--output_dir {args.output_dir}"
        )
        
        if args.username:
            pdsh_command = f"pdsh -R ssh -w {args.username}@{hostname} '{command}'\n"
        else:
            pdsh_command = f"pdsh -R ssh -w {hostname} '{command}'\n"
        
        script_content += pdsh_command
        rank_start += rank_count

    script_filename = "run_all_hosts.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    os.chmod(script_filename, 0o755)  # Make the script executable
    print(f"Generated master shell script: {script_filename}")

# Main function to execute the script
def main():
    parser = argparse.ArgumentParser(description="Generate a shell script to run a Python script on multiple hosts using pdsh.")
    parser.add_argument('--hosts-file', type=str, default="hosts.txt", help="Path to the hosts file (default: hosts.txt).")
    parser.add_argument('--dataset-dir', type=str, default="/mnt/Media/datasets/fineweb-edu", help="Dataset location.")
    parser.add_argument('--output-dir', type=str, default="~/dataset_shard", help="Output shard directory.")
    parser.add_argument('--conda-env', type=str, default="lllm", help="Conda environment name.")
    parser.add_argument('--username', type=str, default=None, help="SSH username.")

    args = parser.parse_args()
    
    # Read hosts from the file
    hosts = read_hosts(args.hosts_file)
    
    # Calculate total ranks in the cluster
    world_size = sum(rank for _, rank in hosts)

    if world_size <= 0:
        print("Error: No hosts found in the hosts file.")
        return

    print(f"World size: {world_size} across {len(hosts)} hosts. Generating master shell script...")

    generate_master_script(hosts, world_size, args)

if __name__ == "__main__":
    main()
