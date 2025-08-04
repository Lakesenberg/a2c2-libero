#!/bin/bash
source ~/.bashrc
cd ~/lerobot && uv venv -p 3.10
cd ~/lerobot && uv pip install --no-cache ".[smolvla]"

# git safe directory
cd ~/lerobot && git config --global --add safe.directory /root/lerobot

cd ~/lerobot

echo "Finished setting up container"

# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null