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

3. **After workspace is ready, start Marimo:**
   ```bash
   ~/start-marimo.sh
   ```

Access Marimo at `http://localhost:8080`

The script automatically clones the [marimo-team/examples](https://github.com/marimo-team/examples) repository with curated example notebooks for data science, machine learning, and AI workflows.

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
4. Installs dependencies from `requirements.txt` (if present)
5. Creates `~/start-marimo.sh` helper script

## Troubleshooting

**Marimo not found:**
```bash
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

**Can't access on port 8080:**
```bash
MARIMO_PORT=8081 ~/start-marimo.sh
```

**Notebooks not loading:**
```bash
# Check if clone succeeded
ls $MARIMO_NOTEBOOKS_DIR

# Check setup logs
cat .brev/logs/setup.log
```

## Resources

- [Marimo Documentation](https://docs.marimo.io)
- [Brev Documentation](https://docs.brev.dev)
