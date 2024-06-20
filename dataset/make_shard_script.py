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

# Function to generate the master shell script using parallel and pdsh
def generate_master_script(hosts, world_size, args):
    script_content = "#!/bin/bash\n"
    script_content += "# Master script to execute shard_dataset.py on multiple hosts using parallel and pdsh\n\n"

    # Set trap to call the kill_remote_jobs function when SIGINT (CTRL+C) is received
    script_content += "\n# Set trap to call the kill_remote_jobs function when SIGINT (CTRL+C) is received\n"
    script_content += "kill_remote_jobs() {\n"
    script_content += f"    pdsh -R ssh -w {','.join([f'{host}' for host, _ in hosts])} 'pkill -f shard_dataset.py'\n"
    script_content += "}\n"
    script_content += "trap 'kill_remote_jobs' SIGINT\n\n"

    commands = []
    rank_start = 0
    for hostname, rank_count in hosts:
        command_parts = [
            f"\"{args.conda_dir}/envs/{args.conda_env}/bin/python\" \"{args.source_dir}/shard_dataset.py\"", 
            f"--dataset-user \"{args.dataset_user}\"",
            f"--dataset-name \"{args.dataset_name}\"",
            f"--rank-start {rank_start}",
            f"--rank-count {rank_count}",
            f"--world-size {world_size}",
            f"--holdout-dir \"{args.holdout_dir}\"",
            f"--holdout-rate {args.holdout_rate}"
        ]
        if args.just_args:
            command_parts.append("--just_args")
        if args.byte_tokens:
            command_parts.append("--byte_tokens")

        command = " ".join(command_parts)

        if args.username:
            pdsh_command = f"pdsh -R ssh -w {args.username}@{hostname} '{command}'"
        else:
            pdsh_command = f"pdsh -R ssh -w {hostname} '{command}'"
        
        commands.append(pdsh_command)
        rank_start += rank_count

    # Write the commands to the script using parallel
    script_content += "parallel --halt-on-error now,fail=1 --lb ::: \\\n"
    script_content += "    \"" + "\" \\\n    \"".join(commands) + "\"\n\n"
    script_content += "trap - SIGINT\n"

    script_filename = "run_all_hosts.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    os.chmod(script_filename, 0o755)  # Make the script executable
    print(f"Generated master shell script: {script_filename}")

# Main function to execute the script
def main():
    parser = argparse.ArgumentParser(description="Generate a shell script to run a Python script on multiple hosts using pdsh.")
    parser.add_argument('--hosts-file', type=str, default="hosts.txt", help="Path to the hosts file (default: hosts.txt).")
    parser.add_argument('--source-dir', type=str, default="~/lllm/dataset", help="Source directory.")
    parser.add_argument('--conda-env', type=str, default="lllm", help="Conda environment name.")
    parser.add_argument('--conda-dir', type=str, default="~/mambaforge", help="Conda environment directory.")
    parser.add_argument('--dataset-user', type=str, default="HuggingFaceFW", help="Dataset user.")
    parser.add_argument('--dataset-name', type=str, default="fineweb-edu", help="Dataset name.")
    parser.add_argument('--output-dir', type=str, default="~/dataset_shard", help="Output shard directory.")
    parser.add_argument('--username', type=str, default=None, help="SSH username.")
    parser.add_argument("--just-args", action="store_true", help="Just write the args file and exit.")

    parser.add_argument("--byte-tokens", action="store_true", help="Tokenize using byte tokens instead of word tokens.")
    parser.add_argument('--holdout-dir', type=str, default="holdout_shard", help="Output directory for holdout set.")
    parser.add_argument('--holdout-rate', type=float, default=0.1, help="Percentage to use for holdout set.")

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
