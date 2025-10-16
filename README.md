# Marimo for GPU Experimentation on Brev

Run interactive Python notebooks with **Marimo** on high-performance **NVIDIA GPU instances** powered by Brev. Perfect for AI/ML experimentation, model training, and data science workflows that require GPU acceleration.

## What is Marimo?

[Marimo](https://marimo.io) is a modern, reactive Python notebook that runs as an interactive web app. Unlike traditional notebooks:

- 🔄 **Reactive** - Cells automatically re-run when dependencies change
- 🐍 **Pure Python** - Notebooks are `.py` files that can be versioned and imported
- 🚀 **Production-ready** - Deploy notebooks as apps with a single command
- 🎨 **Interactive** - Rich UI components and real-time visualizations
- 🔒 **Reproducible** - No hidden state, guaranteed execution order

## Why Marimo + GPU on Brev?

- **Instant GPU Access** - Launch NVIDIA GPU instances in seconds with one click
- **Pre-configured Environment** - Python, CUDA drivers, and ML libraries ready to go
- **Cost Effective** - Pay only for what you use with per-second billing
- **Powerful Hardware** - Access to L40S, A100, H100, and other high-end GPUs
- **Interactive Development** - Experiment with models and visualize results in real-time

Perfect for:
- Training and fine-tuning ML models
- Running inference at scale
- Computer vision and image processing
- LLM experimentation and deployment
- Data analysis with GPU-accelerated libraries

## 🚀 Quick Deploy - GPU Launchables

Deploy Marimo with GPU access instantly using these pre-configured environments:

| GPU Configuration | vRAM | Use Case | Deploy |
|-------------------|------|----------|--------|
| **1x L40S** | 48GB | General ML, Training, Inference | [![Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-34AI5Pvj2cwyqzv50io8WJBpK4t) |

### What's Included

Each deployment includes:
- ✅ Marimo notebook server (running on port 8080)
- ✅ NVIDIA GPU drivers and CUDA toolkit
- ✅ GPU validation notebook
- ✅ Example notebooks from [marimo-team/examples](https://github.com/marimo-team/examples)
- ✅ Pre-installed ML/AI libraries (PyTorch, TensorFlow, etc.)
- ✅ Data science toolkit (pandas, numpy, polars, altair, plotly)
- ✅ No password authentication for ease of use

## Getting Started

### Deploying Your Environment

1. **Choose your GPU configuration** - Click the **Deploy Now** button for your desired environment from the table above
2. **Review and deploy** - On the launchable page, click **Deploy Launchable** 
3. **Sign in** - Create an account or log in to Brev with your email (NVIDIA account required)
4. **Monitor deployment** - Click **Go to Instance Page** to watch your environment spin up
5. **Wait for completion** - Watch for three green status indicators:
   - ✅ **Running** - Instance is live
   - ✅ **Built** - Environment setup complete
   - ✅ **Completed** - Post-install script finished (typically 2-3 minutes)
6. **Access Marimo** - Navigate to the **Access** tab and click the secure link for port 8080
7. **Authenticate** - Log in to Marimo using your Brev account email
8. **Start building** - You're ready to experiment with GPU-accelerated notebooks!

### First Steps

Once inside Marimo:

1. **Validate GPU** - Open `gpu_validation.py` to verify your GPU is detected and working
2. **Run the benchmark** - Click the GPU test button to see CPU vs GPU performance
3. **Explore examples** - Browse the `marimo-examples` directory for inspiration
4. **Create your own** - Click **Create a new notebook** to start experimenting

## Example Notebooks

### GPU Validation (`gpu_validation.py`)
- Check GPU availability and specifications
- Monitor GPU utilization, memory, and temperature
- Visualize GPU performance metrics
- Run basic GPU compute tests

### Marimo Examples
Includes curated notebooks from [marimo-team/examples](https://github.com/marimo-team/examples):
- LLM and AI workflows
- Data visualization
- Interactive dashboards
- SQL and database integration
- And more...

## Advanced Usage

### Service Management

Marimo runs as a systemd service and starts automatically:

```bash
# Check service status
sudo systemctl status marimo

# View logs
sudo journalctl -u marimo -f

# Restart the service
sudo systemctl restart marimo
```

### Customization

Set these environment variables before running the setup script:

| Variable | Description | Default |
|----------|-------------|---------|
| `MARIMO_REPO_URL` | Git repository URL for notebooks | `https://github.com/marimo-team/examples.git` |
| `MARIMO_NOTEBOOKS_DIR` | Directory name for notebooks | `marimo-examples` |
| `MARIMO_PORT` | Port for Marimo server | `8080` |

### Use Your Own Notebooks

```bash
export MARIMO_REPO_URL="https://github.com/your-username/your-notebooks.git"
bash setup.sh
```

### Pre-installed Packages

The environment includes:
- **Data manipulation**: `polars`, `pandas`, `numpy`, `scipy`, `pyarrow`
- **Visualization**: `altair`, `plotly`, `matplotlib`, `seaborn`
- **Machine learning**: `scikit-learn`, `torch`, `tensorflow`
- **AI/LLM**: `openai`, `anthropic`, `instructor`, `openai-whisper`
- **Database**: `marimo[sql]`, `duckdb`, `sqlalchemy`
- **Media processing**: `opencv-python`, `yt-dlp`
- **Utilities**: `requests`, `beautifulsoup4`, `pillow`, `python-dotenv`

## Troubleshooting

### Service Issues
```bash
# Check service status
sudo systemctl status marimo

# View logs
sudo journalctl -u marimo -n 50

# Restart
sudo systemctl restart marimo
```

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Verify PyTorch GPU access
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Can't Access Marimo
- Ensure port 8080 is open in your firewall
- Check if marimo is running: `sudo systemctl status marimo`
- View logs for errors: `sudo journalctl -u marimo -f`

## Manual Setup

If you want to use this setup script in your own repo:

```bash
# Download the setup script
curl -O https://raw.githubusercontent.com/brevdev/marimo/main/setup.sh
chmod +x setup.sh

# Run it
bash setup.sh
```

## Resources

- [Marimo Documentation](https://docs.marimo.io)
- [Marimo Examples](https://github.com/marimo-team/examples)
- [Brev Documentation](https://docs.brev.dev)
- [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/)

## Contributing

Have ideas for improving this setup or want to add more GPU examples? Contributions are welcome! Open an issue or submit a PR at [github.com/brevdev/marimo](https://github.com/brevdev/marimo).
