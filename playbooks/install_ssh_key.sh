#!/bin/bash

KEY_FILE=~/.ssh/id_ed25519

if [ ! -f "$KEY_FILE" ]; then
    echo "Generating new SSH key pair..."
    ssh-keygen -t ed25519 -f "$KEY_FILE" -N ""
    echo "New SSH key pair generated."
else
    echo "SSH key pair already exists."
fi

# specify the inventory file location
inventory_file='inventory.ini'

# parse the hostnames from the inventory file
hosts=$(grep -E '^[^ ]+ ansible_host=' $inventory_file | awk '{print $2}' | awk -F= '{print $2}')

# loop through the hostnames and run the ssh command
for host in $hosts; do
    echo "Adding $host..."
    ssh-keyscan $host >> ~/.ssh/known_hosts
    ssh-copy-id -i $KEY_FILE.pub $USER@$host
done
