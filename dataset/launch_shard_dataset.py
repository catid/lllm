import argparse
import paramiko
import concurrent.futures

# Function to execute SSH command on a host
def execute_ssh_command(hostname, rank_start, rank_count, world_size, args):
    try:
        # Initialize SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the host
        ssh.connect(hostname, username=args.username)

        # Build the command
        command = (
            f"~/mambaforge/envs/{args.conda_env}/bin/python ~/lllm/dataset/shard_dataset.py --dataset_dir {args.dataset_dir} --rank_start {rank_start} --rank_count {rank_count} --world_size {world_size} --output_dir {args.output_dir}"
        )
        
        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # Read output and error streams
        stdout_str = stdout.read().decode()
        stderr_str = stderr.read().decode()
        
        # Close the connection
        ssh.close()
        
        # Report success or failure
        if stderr_str:
            return f"Failed to execute on {hostname} (rank {rank_count}): {stderr_str}"
        else:
            return f"Successfully executed on {hostname} (rank {rank_count}): {stdout_str}"
    
    except Exception as e:
        return f"Error connecting to {hostname} (rank {rank_count}): {str(e)}"

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

# Main function to execute the script
def main():
    parser = argparse.ArgumentParser(description="Run a Python script on multiple hosts via SSH.")
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

    print(f"World size: {world_size} across {len(hosts)} hosts.  Launching `shard_dataset.py` on each host...")

    # Use a thread pool to execute SSH commands in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        future_to_host = {}
        rank_start = 0
        for host, rank_count in hosts:
            future = executor.submit(execute_ssh_command, rank_start, rank_count, world_size, args)

            rank_start += rank_count

            future_to_host[future] = (host, rank_count)

        for future in concurrent.futures.as_completed(future_to_host):
            host, rank_count = future_to_host[future]
            try:
                result = future.result()
                print(f"Host {host}:{rank_count} completed with result: {result}")
            except Exception as exc:
                print(f"{host}:{rank_count} generated an exception: {exc}")

if __name__ == "__main__":
    main()
