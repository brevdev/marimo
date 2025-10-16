# Brev Startup Scripts Guide

## Overview

Startup scripts in Brev allow you to automatically configure and set up your development environment when creating or resetting workspaces. This feature ensures that all dependencies, packages, and tools are consistently installed across all developers working on the same project.

## What Are Startup Scripts?

Startup scripts are bash scripts that run automatically when a Brev workspace is created or reset. They execute after your project repository is cloned and allow you to:

- Install required software and dependencies
- Configure environment variables
- Set up development tools
- Run custom initialization commands
- Standardize development environments across team members

## How Startup Scripts Work

### Execution Context

- **Working Directory**: `/home/ubuntu/<PROJECT_FOLDER_NAME>` (or `/home/brev/<PROJECT_FOLDER_NAME>`)
- **User**: Runs as the workspace user (typically `ubuntu` or `brev`)
- **Timing**: Executes after the git repository is cloned but before the workspace is marked as "Ready"
- **Logs**: Output is saved to `./.brev/logs/setup.log` with archives in `./.brev/logs/archive/`

### Important Notes

⚠️ **Warning**: Do not run long-running or open-ended processes (like `npm start` or web servers) in startup scripts, as this will cause the workspace initialization to hang indefinitely.

## Using Startup Scripts

There are several ways to configure startup scripts for your Brev workspaces:

### Method 1: Commit `.brev/setup.sh` to Your Repository

The most common approach is to commit a setup script directly to your project repository.

1. Create the directory and file:
   ```bash
   mkdir -p .brev
   touch .brev/setup.sh
   chmod +x .brev/setup.sh
   ```

2. Add your setup commands to `.brev/setup.sh`

3. Commit and push to your repository:
   ```bash
   git add .brev/setup.sh
   git commit -m "Add Brev setup script"
   git push
   ```

4. Start a new workspace:
   ```bash
   brev start https://github.com/your-org/your-repo
   ```

Brev will automatically detect and execute `.brev/setup.sh` when creating the workspace.

### Method 2: Use CLI Flags with Setup Script URL

You can specify a setup script URL (like a GitHub Gist) when creating a workspace:

```bash
brev start https://github.com/your-org/your-repo \
  --setup-script https://gist.githubusercontent.com/username/gist-id/raw/setup.sh
```

### Method 3: Use a Separate Setup Repository

Store your setup scripts in a dedicated repository and reference them:

```bash
brev start https://github.com/your-org/your-repo \
  --setup-repo https://github.com/your-org/setup-scripts \
  --setup-path .brev/setup.sh
```

This is useful for:
- Sharing setup scripts across multiple projects
- Keeping setup configuration private while the main project is public
- Managing complex multi-file setup configurations

### Method 4: Specify Custom Path in Repository

If your setup script is located at a custom path within your main repository:

```bash
brev start https://github.com/your-org/your-repo \
  --setup-path scripts/dev-setup.sh
```

### Method 5: Use API/SDK

When creating workspaces programmatically, you can specify startup scripts via the API:

```json
{
  "name": "my-workspace",
  "gitRepo": "github.com:your-org/your-repo.git",
  "startupScript": "#!/bin/bash\necho 'Setting up workspace'\n...",
  "startupScriptPath": ".brev/setup.sh"
}
```

## CLI Commands

### Create Workspace with Startup Script

```bash
# From a Git URL
brev start https://github.com/your-org/your-repo

# From local directory (will look for .brev/setup.sh)
brev start .

# Empty workspace with setup script URL
brev start --name my-workspace \
  --setup-script https://raw.githubusercontent.com/your-org/scripts/main/setup.sh
```

### Reset Workspace (Re-runs Startup Script)

```bash
# Reset workspace and re-run setup script
brev reset <workspace-name>

# Reset to a different branch
brev reset <workspace-name> --branch feature-branch
```

### Recreate Workspace

```bash
# Completely recreate workspace (includes running startup script)
brev recreate <workspace-name>
```

## Personal Terminal Settings

You can also configure personal setup scripts that run on ALL your workspaces:

1. Go to Brev Console → Account Settings → CLI → Terminal Settings
2. Set your personal setup repository URL
3. Set the script path (default: `.brev/setup.sh`)

This is useful for personal configurations like:
- Shell customizations (zsh plugins, aliases)
- Git configuration
- Editor preferences
- Personal tools and utilities

**Note**: Keep personal settings separate from project setup scripts to avoid imposing your preferences on other team members.

## Example Startup Scripts

### Example 1: Basic Node.js Project

```bash
#!/bin/bash

set -euo pipefail

####################################################################################
##### Node.js Project Setup
####################################################################################

##### Node v18.x + npm #####
(echo ""; echo "##### Installing Node.js #####"; echo "";)
sudo apt update
sudo apt install -y ca-certificates curl gnupg
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

##### Yarn #####
(echo ""; echo "##### Installing Yarn #####"; echo "";)
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install -y yarn

##### Install Dependencies #####
(echo ""; echo "##### Installing project dependencies #####"; echo "";)
npm install

##### Setup Environment #####
(echo ""; echo "##### Setting up environment #####"; echo "";)
cp .env.example .env

(echo ""; echo "##### Setup complete! #####"; echo "";)
```

### Example 2: Python Data Science Project

```bash
#!/bin/bash

set -euo pipefail

####################################################################################
##### Python Data Science Project Setup
####################################################################################

##### Update system packages #####
(echo ""; echo "##### Updating system packages #####"; echo "";)
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-distutils

##### Install Python Poetry #####
(echo ""; echo "##### Installing Poetry #####"; echo "";)
curl -sSL https://install.python-poetry.org | python3 -

##### Add Poetry to PATH #####
echo "" >> ~/.bashrc
echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
echo "" >> ~/.zshrc
echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"

##### Install project dependencies #####
(echo ""; echo "##### Installing Python dependencies #####"; echo "";)
poetry install

##### Install Jupyter #####
(echo ""; echo "##### Installing Jupyter #####"; echo "";)
pip3 install jupyter jupyterlab

##### Setup complete #####
(echo ""; echo "##### Setup complete! Run 'poetry shell' to activate environment #####"; echo "";)
```

### Example 3: Go Project

```bash
#!/bin/bash

set -euo pipefail

####################################################################################
##### Go Project Setup
####################################################################################

##### Install Go 1.21 #####
(echo ""; echo "##### Installing Go 1.21 #####"; echo "";)
GO_VERSION="1.21.0"
wget "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz"
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
rm "go${GO_VERSION}.linux-amd64.tar.gz"

##### Update PATH for Go #####
echo "" >> ~/.bashrc
echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.bashrc
echo "export GOPATH=\$HOME/go" >> ~/.bashrc
echo "export PATH=\$PATH:\$GOPATH/bin" >> ~/.bashrc

echo "" >> ~/.zshrc
echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.zshrc
echo "export GOPATH=\$HOME/go" >> ~/.zshrc
echo "export PATH=\$PATH:\$GOPATH/bin" >> ~/.zshrc

export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

##### Install Go tools #####
(echo ""; echo "##### Installing Go tools #####"; echo "";)
go install golang.org/x/tools/gopls@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

##### Download dependencies #####
(echo ""; echo "##### Downloading Go dependencies #####"; echo "";)
go mod download

(echo ""; echo "##### Setup complete! #####"; echo "";)
```

### Example 4: Full-Stack Project (Node + Python + PostgreSQL)

```bash
#!/bin/bash

set -euo pipefail

####################################################################################
##### Full-Stack Application Setup
####################################################################################

##### Install Node.js #####
(echo ""; echo "##### Installing Node.js #####"; echo "";)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

##### Install Python packages #####
(echo ""; echo "##### Installing Python #####"; echo "";)
sudo apt-get install -y python3-pip python3-venv

##### Install PostgreSQL #####
(echo ""; echo "##### Installing PostgreSQL #####"; echo "";)
sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl start postgresql

##### Install Redis #####
(echo ""; echo "##### Installing Redis #####"; echo "";)
sudo apt-get install -y redis-server
sudo systemctl start redis-server

##### Setup Frontend #####
(echo ""; echo "##### Setting up frontend #####"; echo "";)
cd frontend
npm install
cd ..

##### Setup Backend #####
(echo ""; echo "##### Setting up backend #####"; echo "";)
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

##### Setup Database #####
(echo ""; echo "##### Setting up database #####"; echo "";)
sudo -u postgres createdb myapp_dev || echo "Database already exists"

##### Copy environment files #####
cp .env.example .env

(echo ""; echo "##### Setup complete! #####"; echo "";)
(echo ""; echo "Run 'cd frontend && npm start' for frontend"; echo "";)
(echo ""; echo "Run 'cd backend && source venv/bin/activate && python manage.py runserver' for backend"; echo "";)
```

### Example 5: Rust Project

```bash
#!/bin/bash

set -euo pipefail

####################################################################################
##### Rust Project Setup
####################################################################################

##### Install Rust #####
(echo ""; echo "##### Installing Rust #####"; echo "";)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

##### Source Rust environment #####
source "$HOME/.cargo/env"

##### Add to shell configs #####
echo "" >> ~/.bashrc
echo "source \"\$HOME/.cargo/env\"" >> ~/.bashrc
echo "" >> ~/.zshrc
echo "source \"\$HOME/.cargo/env\"" >> ~/.zshrc

##### Install common Rust tools #####
(echo ""; echo "##### Installing Rust tools #####"; echo "";)
cargo install cargo-watch
cargo install cargo-edit
rustup component add rustfmt clippy

##### Build project #####
(echo ""; echo "##### Building project #####"; echo "";)
cargo build

(echo ""; echo "##### Setup complete! #####"; echo "";)
```

### Example 6: Docker-Based Project

```bash
#!/bin/bash

set -euo pipefail

####################################################################################
##### Docker Project Setup
####################################################################################

##### Install Docker dependencies #####
(echo ""; echo "##### Installing Docker #####"; echo "";)
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

##### Add Docker's official GPG key #####
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

##### Set up Docker repository #####
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

##### Install Docker Engine #####
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

##### Add user to docker group #####
sudo usermod -aG docker $USER

##### Install Docker Compose #####
(echo ""; echo "##### Installing Docker Compose #####"; echo "";)
sudo apt-get install -y docker-compose

##### Start containers #####
(echo ""; echo "##### Starting Docker containers #####"; echo "";)
# Note: Need to use 'sg docker' to use docker group in same session
sg docker -c "docker-compose up -d"

(echo ""; echo "##### Setup complete! #####"; echo "";)
(echo ""; echo "You may need to log out and back in for docker group changes to take effect"; echo "";)
```

## Best Practices

### 1. Use Echo Statements for Debugging

Add descriptive echo statements before each major step:

```bash
(echo ""; echo "##### Installing Node.js #####"; echo "";)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
```

This makes logs easier to read and helps identify where errors occur.

### 2. Set Error Handling

Start your script with:

```bash
set -euo pipefail
```

- `set -e`: Exit on error
- `set -u`: Exit on undefined variable
- `set -o pipefail`: Catch errors in pipes

### 3. Update Package Lists

Always update package lists before installing:

```bash
sudo apt-get update
```

### 4. Install Non-Interactive

Use `-y` flag with apt-get to avoid prompts:

```bash
sudo apt-get install -y nodejs
```

### 5. Update Shell Configurations

When modifying PATH or environment variables, update both `.bashrc` and `.zshrc`:

```bash
echo "export PATH=\$PATH:/usr/local/bin" >> ~/.bashrc
echo "export PATH=\$PATH:/usr/local/bin" >> ~/.zshrc
```

### 6. Version Pin Important Dependencies

Specify exact versions for critical dependencies:

```bash
GO_VERSION="1.21.0"
wget "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz"
```

### 7. Test Locally First

Before committing, test your startup script:

```bash
bash .brev/setup.sh
```

### 8. Check Logs

If setup fails, check the logs:

```bash
cat .brev/logs/setup.log
```

### 9. Keep Personal and Project Settings Separate

- **Project scripts** (`.brev/setup.sh` in repo): Dependencies, tools needed by all developers
- **Personal scripts** (configured in Brev Console): Personal preferences, aliases, editor configs

### 10. Document Special Requirements

Add comments explaining non-obvious steps:

```bash
##### Install specific Node version required by project #####
# Note: We use Node 14.x because of dependency compatibility issues
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -
```

## Troubleshooting

### Script Not Running

1. Check if the script file exists and is executable:
   ```bash
   ls -la .brev/setup.sh
   chmod +x .brev/setup.sh
   ```

2. Verify the script path is correct in workspace settings

3. Check workspace logs in Brev Console

### Script Fails Partway Through

1. Review logs at `.brev/logs/setup.log`

2. Add more echo statements to identify exactly where it fails

3. Test the failing command independently

### Workspace Stuck in "Starting" State

This usually means your script is running an infinite process. Check for:
- Long-running servers (`npm start`, `python manage.py runserver`)
- Interactive prompts waiting for input
- Background processes that don't exit

### Dependencies Not Available After Setup

1. Ensure PATH updates are in both `.bashrc` and `.zshrc`

2. Try sourcing the config files:
   ```bash
   source ~/.bashrc
   source ~/.zshrc
   ```

3. Check if you need to export environment variables

### Permission Errors

Use `sudo` for system-level operations:
```bash
sudo apt-get install -y package-name
```

But avoid `sudo` for user-level operations:
```bash
npm install  # Don't use sudo
pip install --user package  # Install for user, not system-wide
```

## Additional Resources

- [Brev CLI Documentation](https://github.com/brevdev/brev-cli)
- [Common Installation Snippets](brev-docs-new/src/pages/docs/how-to/common-installations.md)
- [Brev Discord Community](https://discord.gg/NVDyv7TUgJ)

## Summary

Startup scripts are a powerful feature in Brev that enables:
- **Consistency**: Same environment setup for all team members
- **Automation**: No manual setup steps
- **Reproducibility**: Easy to recreate or reset workspaces
- **Flexibility**: Support for any bash commands or installation procedures

By properly configuring startup scripts, you can ensure that every developer on your team has the exact same development environment, reducing "it works on my machine" problems and speeding up onboarding.

