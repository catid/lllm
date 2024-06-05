# Ansible Playbooks

This directory contains Ansible playbooks for setting up the LLLM training environment on a compute cluster.

Software will be installed under `~/lllm` by default on each machine.


## SSH Setup

Modify the `inventory.ini` file to point to the hosts you are using for training.

Run the script to generate a SSH key and push it to the remote machines.  This will require that you enter a password for each machine.

```bash
cd playbooks

./install_ssh_keys.sh
```

This key will be used to log into the machines in the following steps.


## Ansible Setup

```bash
cd playbooks

# Set up Conda
ansible-playbook install_conda.yml

# Set up the LLLM repo and dependencies
ansible-playbook setup_lllm.yml
```
