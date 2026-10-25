# SSH tunnel

Colab forwards local ports to a relay server with a reverse SSH tunnel (`ssh -R`) as the `tunnel` user. Your machine
then forwards the same ports from the relay with `connect.sh` (`ssh -L`). Ports are bound to `127.0.0.1` on the relay,
so they are only reachable through SSH.

ComfyUI uses port 8188, started by `ssh_tunnel.py`.

## Adding a port

1. **Relay:** allow the `tunnel` key to listen on the port. Each port needs its own `permitlisten` entry on the key's
   line in `/home/tunnel/.ssh/authorized_keys`:

   ```
   restrict,port-forwarding,permitlisten="127.0.0.1:8188",permitlisten="127.0.0.1:6006" ssh-ed25519 AAA... colab
   ```

   The file is read on each login, so no restart is needed. Without this entry, the Colab log shows
   `Error: remote port forwarding failed for listen port <port>`.

2. **Colab:** start a reverse tunnel for the port, following `ssh_tunnel.py` with `PORT` changed. Give it its own log
   file so it does not overwrite `/content/ssh_tunnel.log`.

3. **Your machine:** pass every port to `connect.sh`:

   ```
   colab/connect.sh you@<relay> ~/.ssh/key.pem 8188 6006
   ```

   Each port is served at `http://localhost:<port>`.

## Troubleshooting

If the forward still fails after the relay allows the port, something else already holds it:

- `pgrep -af ssh_tunnel` in Colab: more than one loop for the same port means the cell ran twice. Stop the extra one
  with `pkill -f "127.0.0.1:<port>:localhost"`.
- `sudo ss -ltnp | grep <port>` on the relay: an `sshd` entry is usually an old session that is freed when it times out.
  Another program on the port means picking a different port.
