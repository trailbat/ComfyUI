#!/usr/bin/env bash
# Connects to ports that colab/ssh_tunnel.py (or a similar reverse tunnel) forwards to the relay, and serves each at
# http://localhost:<port> until Ctrl+C. Ports default to 8188 (ComfyUI). Runs in Git Bash or any shell with OpenSSH:
#   colab/connect.sh ubuntu@<relay ip> ~/.ssh/lightsail.pem [port...]
# The key can be left empty ("") when ~/.ssh/config already has one for the host.
set -euo pipefail

host=${1:?usage: connect.sh user@relay [key] [port...]}
key=${2:-}
ports=("${@:3}")
[ ${#ports[@]} -eq 0 ] && ports=(8188)

forwards=()
for port in "${ports[@]}"; do
    forwards+=(-L "$port:127.0.0.1:$port")
    echo "http://localhost:$port"
done

echo "Ctrl+C to disconnect"
exec ssh -N ${key:+-i "$key"} -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes "${forwards[@]}" "$host"
