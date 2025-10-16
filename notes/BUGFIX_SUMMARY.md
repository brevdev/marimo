# Marimo Setup Script - Bug Fix Summary

## Problem Identified

The Marimo setup script was failing when run in Brev VM Mode with this error:

```
/opt/oncreate_lifecycle_script.sh: line 23: HOME: unbound variable
[ERROR] Service oncreate-lifecycle-script setup failed
```

### Root Cause

The script used `set -euo pipefail` which causes immediate exit on unbound variables. When Brev runs the script via systemd service, the `$HOME` environment variable was not set, causing the script to fail at line 23 when trying to append to `~/.bashrc`.

**Result:** Marimo was successfully installed, but the script crashed before:
- Cloning the marimo-team/examples repository
- Installing dependencies from requirements.txt
- Creating the `~/start-marimo.sh` helper script

## Solution Implemented

Updated `.brev/setup.sh` with the following fixes:

### 1. Set default environment variables
```bash
# Set HOME if not defined (for systemd service context)
HOME="${HOME:-/home/ubuntu}"
USER="${USER:-ubuntu}"
```

### 2. Made shell config updates more robust
```bash
# Added error suppression and fallback
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" 2>/dev/null || true
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
```

### 3. Used explicit paths throughout
```bash
# Changed from relative paths to explicit $HOME-based paths
cd "$HOME"
git clone "$REPO_URL" "$NOTEBOOKS_DIR"
cat > "$HOME/start-marimo.sh" << 'EOF'
```

### 4. Added ownership fixes for root execution
```bash
# Fix ownership if running as root
if [ "$(id -u)" -eq 0 ] && [ -n "$USER" ]; then
    chown -R "$USER:$USER" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/start-marimo.sh"
    if [ -d "$HOME/$NOTEBOOKS_DIR" ]; then
        chown -R "$USER:$USER" "$HOME/$NOTEBOOKS_DIR"
    fi
fi
```

## Testing the Fix

### Option 1: Test in current workspace (marimo-test-1)

You can manually run the fixed script on the existing workspace:

```bash
# SSH into the workspace
brev shell marimo-test-1

# Download and run the fixed script
curl -o setup.sh https://raw.githubusercontent.com/brevdev/marimo/main/.brev/setup.sh
chmod +x setup.sh
bash setup.sh

# Verify it worked
ls -la ~/marimo-examples/
ls -la ~/start-marimo.sh
~/start-marimo.sh
```

### Option 2: Create a new Brev workspace from GitHub repo

```bash
# Create a new workspace from the GitHub repo
brev start https://github.com/brevdev/marimo

# Wait for workspace to be ready, then SSH in
brev shell <workspace-name>

# The setup should have run automatically
# Verify it worked
ls -la ~/marimo-examples/
ls -la ~/start-marimo.sh

# Start Marimo
~/start-marimo.sh
```

### Option 3: Create a new VM Mode workspace with the fixed script

Copy the fixed script from `.brev/setup.sh` and paste it when creating a new VM Mode workspace in the Brev Console.

## Files Changed

- `.brev/setup.sh` - Fixed the setup script with proper environment variable handling
- `README.md` - Updated with complete documentation (resolved merge conflict)
- `STARTUP_SCRIPT_GUIDE.md` - Comprehensive guide for Brev startup scripts

## Verification Checklist

After running the fixed script, verify:

- [ ] Marimo is installed: `marimo --version` returns `0.17.0`
- [ ] Examples cloned: `ls ~/marimo-examples/` shows notebook files
- [ ] Helper script created: `ls ~/start-marimo.sh` exists and is executable
- [ ] PATH updated: `which marimo` returns `/usr/local/bin/marimo` or `~/.local/bin/marimo`
- [ ] Marimo runs: `~/start-marimo.sh` starts Marimo server on port 8080

## Next Steps

1. **Test the fix** using one of the methods above
2. **Delete the old workspace** if you create a new one: `brev delete marimo-test-1`
3. **Update documentation** if needed based on real-world usage
4. **Consider adding** error logging to capture future issues

## Related Resources

- [Brev Startup Scripts Documentation](STARTUP_SCRIPT_GUIDE.md)
- [Marimo Documentation](https://docs.marimo.io)
- [GitHub Repository](https://github.com/brevdev/marimo)

