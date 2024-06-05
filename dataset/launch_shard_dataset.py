import argparse
import paramiko
import concurrent.futures

# Function to execute SSH command on a host
def execute_ssh_command(hostname, rank_count, total_ranks, args):
    try:
        # Initialize SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the host
        ssh.connect(hostname, username=args.username)

        # Build the command
        command = (
            f"~/mambaforge/envs/{args.conda_env}/bin/python ~/lllm/dataset/shard_dataset.py {total_ranks} {rank_count} {args.dataset_location}"
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
    parser.add_argument('--dataset-location', type=str, default="/mnt/Media/datasets/fineweb-edu", help="Dataset location.")
    parser.add_argument('--conda-env', type=str, default="lllm", help="Conda environment name.")
    parser.add_argument('--username', type=str, default=None, help="SSH username.")

    args = parser.parse_args()
    
    # Read hosts from the file
    hosts = read_hosts(args.hosts_file)
    
    # Calculate total ranks in the cluster
    total_ranks = sum(rank for _, rank in hosts)

    if total_ranks <= 0:
        print("Error: No hosts found in the hosts file.")
        return

    print(f"Total ranks: {total_ranks} across {len(hosts)} hosts.  Launching `shard_dataset.py` on each host...")

    # Use a thread pool to execute SSH commands in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as executor:
        future_to_host = {executor.submit(execute_ssh_command, host, rank, total_ranks, args): (host, rank) for host, rank in hosts}

        for future in concurrent.futures.as_completed(future_to_host):
            host, rank = future_to_host[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"{host} (rank {rank}) generated an exception: {exc}")

if __name__ == "__main__":
    main()
