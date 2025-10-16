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

##### Fix ownership if running as root #####
if [ "$(id -u)" -eq 0 ] && [ -n "$USER" ]; then
    (echo ""; echo "##### Fixing file ownership #####"; echo "";)
    chown -R "$USER:$USER" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/start-marimo.sh" 2>/dev/null || true
    if [ -d "$HOME/$NOTEBOOKS_DIR" ]; then
        chown -R "$USER:$USER" "$HOME/$NOTEBOOKS_DIR" 2>/dev/null || true
    fi
fi

(echo ""; echo "##### Setup complete! #####"; echo "";)
(echo "To start Marimo, run: ~/start-marimo.sh"; echo "";)
