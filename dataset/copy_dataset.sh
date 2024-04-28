#!/bin/bash

# This is where the data is stored
SOURCE_HOST="gpu4.lan"
DIR_PATH="~/lllm/dataset"

# Read the list of hosts from the inventory file
mapfile -t HOSTS < copy_hosts.txt

function rsync_to_hosts() {
  local source_host="$1"
  local local_path="$2"
  local remote_path="$3"
  shift 3
  local hosts=("$@")

  echo "Starting parallel rsync transfers..."

  # Define the rsync command to run in parallel
  rsync_command() {
    local src_host="$1"
    local host="$2"
    local local_p="$3"
    local remote_p="$4"
    if [[ "$host" != "$src_host" ]]; then
        trap 'kill -INT $!' INT
        ssh "$src_host" "rsync -az --info=progress2 -B 8192 -e 'ssh -c aes128-ctr' '$local_p' '$host:$remote_p'"
        echo "Rsync from $src_host to $host completed."
    fi
  }

  export -f rsync_command

  # Run the rsync command in parallel with 2 simultaneous jobs
  parallel --halt-on-error 1 --progress -j 3 rsync_command "$source_host" {} "$local_path" "$remote_path" ::: "${hosts[@]}"
}

# Call the function with the required variables
rsync_to_hosts "$SOURCE_HOST" "$LOCAL_PATH" "$REMOTE_PATH" "${HOSTS[@]}"
