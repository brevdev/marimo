#!/bin/bash

set -euo pipefail

####################################################################################
##### Marimo Setup for Brev
####################################################################################
# Defaults to cloning marimo-team/examples repository
# Set MARIMO_REPO_URL to use your own notebooks repository
# Set MARIMO_REPO_URL="" to skip cloning entirely
####################################################################################

# Set HOME if not defined (for systemd service context)
HOME="${HOME:-/home/ubuntu}"
USER="${USER:-ubuntu}"

REPO_URL="${MARIMO_REPO_URL:-https://github.com/marimo-team/examples.git}"
NOTEBOOKS_DIR="${MARIMO_NOTEBOOKS_DIR:-marimo-examples}"

##### Install Marimo #####
(echo ""; echo "##### Installing Marimo #####"; echo "";)
pip3 install --upgrade marimo

##### Add to PATH #####
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" 2>/dev/null || true
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

##### Clone notebooks if URL provided #####
if [ -n "$REPO_URL" ]; then
    (echo ""; echo "##### Cloning notebooks from $REPO_URL #####"; echo "";)
    cd "$HOME"
    git clone "$REPO_URL" "$NOTEBOOKS_DIR"
    
    # Install dependencies if requirements.txt exists
    if [ -f "$HOME/$NOTEBOOKS_DIR/requirements.txt" ]; then
        (echo ""; echo "##### Installing dependencies #####"; echo "";)
        pip3 install -r "$HOME/$NOTEBOOKS_DIR/requirements.txt"
    fi
fi

##### Create helper script to run Marimo #####
(echo ""; echo "##### Creating start-marimo.sh helper script #####"; echo "";)
cat > "$HOME/start-marimo.sh" << 'EOF'
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/${MARIMO_NOTEBOOKS_DIR:-marimo-examples}" 2>/dev/null || cd "$HOME"
marimo edit --host 0.0.0.0 --port ${MARIMO_PORT:-8080} --headless
EOF
chmod +x "$HOME/start-marimo.sh"

##### Create systemd service for Marimo #####
(echo ""; echo "##### Setting up Marimo systemd service #####"; echo "";)
sudo tee /etc/systemd/system/marimo.service > /dev/null << EOF
[Unit]
Description=Marimo Notebook Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/$NOTEBOOKS_DIR
Environment="PATH=/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
Environment="HOME=$HOME"
Environment="MARIMO_PORT=${MARIMO_PORT:-8080}"
ExecStart=/usr/local/bin/marimo edit --host 0.0.0.0 --port \${MARIMO_PORT} --headless
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=marimo

[Install]
WantedBy=multi-user.target
EOF

##### Enable and start Marimo service #####
(echo ""; echo "##### Enabling and starting Marimo service #####"; echo "";)
sudo systemctl daemon-reload
sudo systemctl enable marimo.service
sudo systemctl start marimo.service

# Wait a moment for service to start
sleep 3

# Check service status
sudo systemctl status marimo.service --no-pager || true

##### Fix ownership if running as root #####
if [ "$(id -u)" -eq 0 ] && [ -n "$USER" ]; then
    (echo ""; echo "##### Fixing file ownership #####"; echo "";)
    chown -R "$USER:$USER" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/start-marimo.sh" 2>/dev/null || true
    if [ -d "$HOME/$NOTEBOOKS_DIR" ]; then
        chown -R "$USER:$USER" "$HOME/$NOTEBOOKS_DIR" 2>/dev/null || true
    fi
fi

(echo ""; echo "##### Setup complete! #####"; echo "";)
(echo "Marimo is now running as a systemd service on port ${MARIMO_PORT:-8080}"; echo "";)
(echo "Service management commands:"; echo "";)
(echo "  - Check status:  sudo systemctl status marimo"; echo "";)
(echo "  - View logs:     sudo journalctl -u marimo -f"; echo "";)
(echo "  - Restart:       sudo systemctl restart marimo"; echo "";)
(echo "  - Stop:          sudo systemctl stop marimo"; echo "";)
(echo ""; echo "You can also run Marimo manually with: ~/start-marimo.sh"; echo "";)
