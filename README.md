# Marimo Setup for Brev

Automated setup script to install Marimo on Brev with example notebooks from [marimo-team/examples](https://github.com/marimo-team/examples).

## Quick Start

1. **Copy the setup script to your repo:**
   ```bash
   mkdir -p .brev
   cp setup.sh .brev/setup.sh
   git add .brev/setup.sh
   git commit -m "Add Marimo setup"
   git push
   ```

2. **Start a Brev workspace:**
   ```bash
   brev start https://github.com/your-org/your-repo
   ```

3. **After workspace is ready, Marimo will be running automatically!**
   
   Access Marimo at `http://localhost:8080` (or your Brev workspace URL on port 8080)
   
   **Note:** Password authentication is disabled by default for ease of use.

The setup script automatically:
- Installs Marimo
- Clones the [marimo-team/examples](https://github.com/marimo-team/examples) repository with curated example notebooks
- Installs common Python packages for data science and visualization
- Sets up Marimo as a systemd service that starts automatically and restarts on failure

## Configuration

Set these environment variables in your Brev workspace or shell to customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `MARIMO_REPO_URL` | Git repository URL for notebooks | `https://github.com/marimo-team/examples.git` |
| `MARIMO_NOTEBOOKS_DIR` | Directory name for notebooks | `marimo-examples` |
| `MARIMO_PORT` | Port for Marimo server | `8080` |

### Use Your Own Notebooks

To use your own notebooks repository instead of the examples:

```bash
export MARIMO_REPO_URL="https://github.com/your-username/your-notebooks.git"
```

### Skip Cloning Entirely

To install Marimo without cloning any repository:

```bash
export MARIMO_REPO_URL=""
```

## What It Does

The setup script:
1. Installs Marimo via pip
2. Updates PATH in `.bashrc` and `.zshrc`
3. Clones the marimo-team/examples repository (or your custom repo if `MARIMO_REPO_URL` is set)
4. Installs common Python packages for data science and marimo examples:
   - Data manipulation: `polars`, `pandas`, `numpy`, `scipy`, `pyarrow`
   - Visualization: `altair`, `plotly`, `matplotlib`, `seaborn`
   - Machine learning: `scikit-learn`
   - AI/LLM: `openai`, `anthropic`
   - Database: `marimo[sql]`, `duckdb`, `sqlalchemy`
   - Utilities: `requests`, `beautifulsoup4`, `pillow`
5. Installs additional dependencies from `requirements.txt` (if present in the notebooks directory)
6. Creates and starts a systemd service to run Marimo automatically (without password authentication)
7. Creates `~/start-marimo.sh` helper script for manual runs

## Service Management

Marimo runs as a systemd service and starts automatically on boot:

```bash
# Check service status
sudo systemctl status marimo

# View logs in real-time
sudo journalctl -u marimo -f

# Restart the service
sudo systemctl restart marimo

# Stop the service
sudo systemctl stop marimo

# Start the service
sudo systemctl start marimo
```

## Troubleshooting

**Service not running:**
```bash
# Check service status
sudo systemctl status marimo

# View service logs
sudo journalctl -u marimo -n 50

# Restart the service
sudo systemctl restart marimo
```

**Can't access Marimo on port 8080:**
```bash
# Check if marimo is listening
sudo netstat -tlnp | grep 8080
# or
sudo ss -tlnp | grep 8080

# Check service logs for errors
sudo journalctl -u marimo -f
```

**Marimo command not found:**
```bash
# Check installation
which marimo
pip3 list | grep marimo

# Add to PATH manually
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

**Notebooks not loading:**
```bash
# Check if clone succeeded
ls ~/marimo-examples/

# Check setup logs
cat ~/.lifecycle-script-*.log
```

## Resources

- [Marimo Documentation](https://docs.marimo.io)
- [Brev Documentation](https://docs.brev.dev)
