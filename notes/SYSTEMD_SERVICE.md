# Marimo Systemd Service Implementation

## Overview

Updated the Marimo setup script to automatically configure and start Marimo as a systemd service, ensuring it:
- Starts automatically on workspace boot
- Restarts automatically if it crashes
- Runs in the background continuously
- Provides proper logging via systemd journal

## Changes Made

### 1. Setup Script Updates (`.brev/setup.sh`)

Added systemd service configuration that creates `/etc/systemd/system/marimo.service`:

```ini
[Unit]
Description=Marimo Notebook Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/marimo-examples
Environment="PATH=/usr/local/bin:/usr/bin:/bin:/home/ubuntu/.local/bin"
Environment="HOME=/home/ubuntu"
Environment="MARIMO_PORT=8080"
ExecStart=/usr/local/bin/marimo edit --host 0.0.0.0 --port ${MARIMO_PORT} --headless
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=marimo

[Install]
WantedBy=multi-user.target
```

### 2. Service Management

The setup script now:
1. Creates the systemd service file
2. Reloads systemd daemon
3. Enables the service (start on boot)
4. Starts the service immediately
5. Shows service status to confirm it's running

### 3. Documentation Updates

Updated `README.md` to include:
- Information that Marimo starts automatically
- Service management commands
- Updated troubleshooting section with service-specific debugging

## Service Features

### Auto-Start on Boot
- Service is enabled via `systemctl enable marimo`
- Will start automatically when the Brev workspace boots

### Auto-Restart on Failure
- `Restart=always` ensures the service restarts if it crashes
- `RestartSec=10` waits 10 seconds before restarting
- Provides resilience against transient errors

### Proper Logging
- All output goes to systemd journal
- View logs with: `sudo journalctl -u marimo -f`
- Logs persist across restarts

### Environment Configuration
- PATH includes system and user directories
- HOME is set correctly for the ubuntu user
- MARIMO_PORT can be customized via environment variable

## Testing the Service

### On Existing Workspace (kj-marimo-2)

Since the new setup script wasn't used during creation, manually set up the service:

```bash
brev shell kj-marimo-2

# Download and run the updated script
curl -o setup.sh https://raw.githubusercontent.com/brevdev/marimo/main/.brev/setup.sh
chmod +x setup.sh
bash setup.sh
```

### On New Workspace

Create a new workspace from the repo:

```bash
brev start https://github.com/brevdev/marimo
```

The service will be automatically configured and started.

## Service Management Commands

```bash
# Check if service is running
sudo systemctl status marimo

# View live logs
sudo journalctl -u marimo -f

# View last 50 log lines
sudo journalctl -u marimo -n 50

# Restart service
sudo systemctl restart marimo

# Stop service
sudo systemctl stop marimo

# Start service
sudo systemctl start marimo

# Disable auto-start
sudo systemctl disable marimo

# Re-enable auto-start
sudo systemctl enable marimo

# Check if service is enabled
sudo systemctl is-enabled marimo
```

## Troubleshooting

### Service fails to start

```bash
# Check detailed status
sudo systemctl status marimo -l

# View recent logs
sudo journalctl -u marimo -n 100

# Check if marimo binary exists
which marimo
ls -la /usr/local/bin/marimo

# Check working directory exists
ls -la /home/ubuntu/marimo-examples/
```

### Port already in use

```bash
# Check what's using port 8080
sudo ss -tlnp | grep 8080

# If needed, kill the process
sudo kill <PID>

# Or change the port in service file
sudo nano /etc/systemd/system/marimo.service
# Update MARIMO_PORT value
sudo systemctl daemon-reload
sudo systemctl restart marimo
```

### Permission issues

```bash
# Check file ownership
ls -la /home/ubuntu/marimo-examples/

# Fix ownership if needed
sudo chown -R ubuntu:ubuntu /home/ubuntu/marimo-examples/

# Restart service
sudo systemctl restart marimo
```

### Service won't stay running

```bash
# Watch logs in real-time to see errors
sudo journalctl -u marimo -f

# Check if there are Python dependency issues
cd /home/ubuntu/marimo-examples/
pip3 list | grep marimo

# Test marimo manually
marimo edit --host 0.0.0.0 --port 8080 --headless
```

## Advantages Over Manual Execution

1. **Automatic startup**: No need to remember to start Marimo after workspace creation
2. **Resilience**: Automatically restarts on crash or error
3. **Background execution**: Runs as a daemon, doesn't require an active terminal
4. **Logging**: Centralized logging via systemd journal
5. **Service management**: Standard systemd commands for all operations
6. **Clean integration**: Works with Brev's workspace lifecycle

## Future Enhancements

Potential improvements to consider:

1. **Multi-instance support**: Run multiple Marimo servers on different ports
2. **Resource limits**: Add memory/CPU limits via systemd
3. **Health checks**: Implement endpoint monitoring
4. **Log rotation**: Configure log size limits
5. **Environment file**: Move configuration to `/etc/default/marimo`
6. **Graceful shutdown**: Add pre-stop script to save state
7. **User isolation**: Consider running in a Python venv

## Related Files

- `.brev/setup.sh` - Main setup script with systemd configuration
- `README.md` - User documentation
- `/etc/systemd/system/marimo.service` - Service definition (created on workspace)
- `~/start-marimo.sh` - Manual execution script (still available)

## References

- [systemd.service documentation](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [Marimo documentation](https://docs.marimo.io)
- [Brev startup scripts guide](../STARTUP_SCRIPT_GUIDE.md)

