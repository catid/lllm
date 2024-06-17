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

# Function to generate the master shell script for torchrun
def generate_master_script(hosts, world_size, args):
    script_content = "#!/bin/bash\n"
    script_content += "# Master script to execute training on multiple hosts using torchrun\n\n"

    # Set the MASTER_ADDR and MASTER_PORT
    master_addr = hosts[0][0]
    master_port = args.master_port

    script_content += f"export MASTER_ADDR={master_addr}\n"
    script_content += f"export MASTER_PORT={master_port}\n\n"

    commands = []
    rank_start = 0
    for hostname, rank_count in hosts:
        command_parts = [
            f"torchrun --nproc_per_node={rank_count}",
            f"--nnodes={len(hosts)}",
            f"--node_rank={rank_start // rank_count}",
            f"--master_addr={master_addr}",
            f"--master_port={master_port}",
            f"{args.script} {args.script_args}"
        ]

        command = " ".join(command_parts)

        if args.username:
            ssh_command = f"ssh {args.username}@{hostname} '{command}'"
        else:
            ssh_command = f"ssh {hostname} '{command}'"
        
        commands.append(ssh_command)
        rank_start += rank_count

    # Write the commands to the script
    script_content += "\n".join(commands) + "\n"

    script_filename = "run_all_hosts.sh"
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    os.chmod(script_filename, 0o755)  # Make the script executable
    print(f"Generated master shell script: {script_filename}")

# Main function to execute the script
def main():
    parser = argparse.ArgumentParser(description="Generate a shell script to run a Python script on multiple hosts using torchrun.")
    parser.add_argument('--hosts-file', type=str, required=True, help="Path to the hosts file.")
    parser.add_argument('--script', type=str, required=True, help="Path to the training script.")
    parser.add_argument('--script-args', type=str, default="", help="Arguments to pass to the training script.")
    parser.add_argument('--master-port', type=int, default=12355, help="Port number for master node.")
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