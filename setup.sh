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

##### Install Python and pip if not available #####
if ! command -v pip3 &> /dev/null; then
    (echo ""; echo "##### Installing Python and pip3 #####"; echo "";)
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

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
    git clone "$REPO_URL" "$NOTEBOOKS_DIR" 2>/dev/null || echo "Repository already exists"
    
    # Install PyTorch with CUDA support first (large download)
    (echo ""; echo "##### Installing PyTorch with CUDA support #####"; echo "";)
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install common packages for marimo examples
    (echo ""; echo "##### Installing common packages for marimo examples #####"; echo "";)
    pip3 install --no-cache-dir --upgrade \
        polars altair plotly pandas numpy scipy scikit-learn \
        matplotlib seaborn pyarrow openai anthropic requests \
        beautifulsoup4 pillow 'marimo[sql]' duckdb sqlalchemy \
        instructor mohtml openai-whisper opencv-python python-dotenv \
        wigglystuff yt-dlp psutil pynvml GPUtil
    
    # Install dependencies if requirements.txt exists
    if [ -f "$HOME/$NOTEBOOKS_DIR/requirements.txt" ]; then
        (echo ""; echo "##### Installing additional dependencies from requirements.txt #####"; echo "";)
        pip3 install -r "$HOME/$NOTEBOOKS_DIR/requirements.txt"
    fi
    
    # Install gpu-burn for hardcore GPU stress testing (optional but recommended)
    (echo ""; echo "##### Installing gpu-burn for advanced GPU stress testing #####"; echo "";)
    cd "$HOME"
    if [ ! -d "gpu-burn" ]; then
        git clone https://github.com/wilicc/gpu-burn.git 2>/dev/null || true
        if [ -d "gpu-burn" ]; then
            cd gpu-burn
            make 2>/dev/null || echo "Note: gpu-burn compilation skipped (requires CUDA development toolkit)"
            cd "$HOME"
        fi
    fi
    
    # Copy GPU validation notebook and other custom notebooks from this repo
    (echo ""; echo "##### Adding GPU validation notebook to notebooks directory #####"; echo "";)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Try multiple possible locations for gpu_validation.py
    if [ -f "$SCRIPT_DIR/gpu_validation.py" ]; then
        cp "$SCRIPT_DIR/gpu_validation.py" "$HOME/$NOTEBOOKS_DIR/gpu_validation.py"
    elif [ -f "$HOME/marimo/gpu_validation.py" ]; then
        cp "$HOME/marimo/gpu_validation.py" "$HOME/$NOTEBOOKS_DIR/gpu_validation.py"
    elif [ -f "/workspace/gpu_validation.py" ]; then
        cp /workspace/gpu_validation.py "$HOME/$NOTEBOOKS_DIR/gpu_validation.py"
    fi
fi

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
ExecStart=/usr/local/bin/marimo edit --host 0.0.0.0 --port \${MARIMO_PORT} --headless --no-token
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=marimo

[Install]
WantedBy=multi-user.target
EOF

##### Fix ownership if running as root #####
if [ "$(id -u)" -eq 0 ] && [ -n "$USER" ]; then
    (echo ""; echo "##### Fixing file ownership #####"; echo "";)
    chown -R "$USER:$USER" "$HOME/.bashrc" "$HOME/.zshrc" 2>/dev/null || true
    if [ -d "$HOME/$NOTEBOOKS_DIR" ]; then
        chown -R "$USER:$USER" "$HOME/$NOTEBOOKS_DIR" 2>/dev/null || true
    fi
fi

##### Enable and start Marimo service #####
(echo ""; echo "##### Enabling and starting Marimo service #####"; echo "";)
sudo systemctl daemon-reload
sudo systemctl enable marimo.service 2>/dev/null || true
sudo systemctl start marimo.service

# Wait for service to start
sleep 2

(echo ""; echo ""; echo "═══════════════════════════════════════════════════════════"; echo "";)
(echo "  ✅  Setup Complete! Marimo is now running"; echo "";)
(echo "═══════════════════════════════════════════════════════════"; echo "";)
(echo ""; echo "📍 Notebooks Location: $HOME/$NOTEBOOKS_DIR"; echo "";)
(echo "🌐 Access URL: http://localhost:${MARIMO_PORT:-8080}"; echo "";)
(echo "🎮 GPU Validation: Open gpu_validation.py to test your GPU"; echo "";)
(echo ""; echo "Useful commands:"; echo "";)
(echo "  • Check status:  sudo systemctl status marimo"; echo "";)
(echo "  • View logs:     sudo journalctl -u marimo -f"; echo "";)
(echo "  • Restart:       sudo systemctl restart marimo"; echo "";)
(echo ""; echo "═══════════════════════════════════════════════════════════"; echo "";)
