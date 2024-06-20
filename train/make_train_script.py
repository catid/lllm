import argparse
import os

# Function to read hosts from the file
def read_hosts(file_path):
    hosts = []
    world_size = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                hostname, rank_count = line.split()
                rank_count = int(rank_count)
                hosts.append((hostname, rank_count))
                world_size += rank_count
    return hosts, world_size

# Function to generate the master shell script using parallel and pdsh
def generate_master_script(hosts, world_size, args, unknown_args):
    script_content = "#!/bin/bash\n"
    script_content += "# Master script to execute train.py on multiple hosts using parallel and pdsh\n"

    # Set trap to call the kill_remote_jobs function when SIGINT (CTRL+C) is received
    script_content += "\n# Set trap to call the kill_remote_jobs function when SIGINT (CTRL+C) is received\n"
    script_content += "kill_remote_jobs() {\n"
    script_content += f"    pdsh -R ssh -w {','.join([f'{host}' for host, _ in hosts])} 'pkill -f train.py'\n"
    script_content += "}\n"
    script_content += "trap 'kill_remote_jobs' SIGINT\n\n"

    commands = []
    rank_start = 0
    for hostname, rank_count in hosts:
        command_parts = [
            f"\"{args.conda_dir}/envs/{args.conda_env}/bin/torchrun\"",
            f"--nproc_per_node={rank_count}",
            f"--nnodes={len(hosts)}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            f"{args.source_dir}/train.py",
        ]
        command_parts = command_parts + unknown_args

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

    script_filename = "launch_train.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    os.chmod(script_filename, 0o755)  # Make the script executable
    print(f"Generated master shell script: {script_filename}")

# Main function to execute the script
def main():
    parser = argparse.ArgumentParser(description="Generate a shell script to run a Python script on multiple hosts using torchrun.")
    parser.add_argument('--hosts-file', type=str, default="hosts.txt", help="Path to the hosts file (default: hosts.txt).")
    parser.add_argument('--source-dir', type=str, default="~/lllm/train", help="Source directory.")
    parser.add_argument('--conda-env', type=str, default="lllm", help="Conda environment name.")
    parser.add_argument('--conda-dir', type=str, default="~/mambaforge", help="Conda environment directory.")
    parser.add_argument('--username', type=str, default=None, help="SSH username.")
    parser.add_argument("--master-addr", type=str, default=None, help="Address of master node. Default: First hosts.txt entry")
    parser.add_argument("--master-port", type=int, default=12345, help="Port to use for the master node")

    # We interpret these args and pass the rest to the train.py script
    args, unknown_args = parser.parse_known_args()

    print(f"Arguments: {args}")
    print(f"Forwarded args: {unknown_args}")

    # Read hosts from the file
    hosts, world_size = read_hosts(args.hosts_file)

    if len(hosts) <= 0:
        print("Error: No hosts found in the hosts file.")
        return

    print(f"World size: {world_size} across {len(hosts)} hosts. Generating master shell script...")

    if args.master_addr is None:
        args.master_addr = hosts[0][0]

    generate_master_script(hosts, world_size, args, unknown_args)

if __name__ == "__main__":
    main()
