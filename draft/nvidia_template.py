"""nvidia_template.py

NVIDIA + Marimo Best Practices Template
=======================================

This template demonstrates:
- Reactive GPU resource monitoring
- Interactive parameter tuning
- Real-time visualization updates
- Production-ready error handling
- NVIDIA-specific optimizations
- Clean, maintainable code structure

Author: Brev.dev Team
Date: 2025-10-17
GPU Requirements: L40S or higher recommended
"""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def __():
    """Import dependencies with automatic package management"""
    import marimo as mo
    import torch
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from typing import Optional, Dict, List, Tuple
    import subprocess
    import json
    from dataclasses import dataclass
    return mo, torch, np, pd, px, go, Optional, Dict, List, Tuple, subprocess, json, dataclass


@app.cell
def __(mo):
    """Configuration Section - Interactive Sliders"""
    mo.md(
        """
        # NVIDIA GPU Demo: [Your Title Here]
        
        **Description**: [Brief description of what this notebook does]
        
        **Requirements**:
        - NVIDIA GPU with compute capability 7.0+
        - CUDA 12.0+
        - 16GB+ GPU memory recommended
        
        ## Configuration
        """
    )
    return


@app.cell
def __(mo):
    """Interactive parameter controls"""
    batch_size = mo.ui.slider(
        start=1, stop=128, step=1, value=32,
        label="Batch Size", show_value=True
    )
    
    precision = mo.ui.dropdown(
        options=["float32", "float16", "bfloat16"],
        value="float16",
        label="Precision"
    )
    
    enable_cudnn_benchmark = mo.ui.checkbox(
        value=True,
        label="Enable cuDNN Benchmark"
    )
    
    mo.hstack([batch_size, precision, enable_cudnn_benchmark], justify="start")
    return batch_size, precision, enable_cudnn_benchmark


@app.cell
def __(torch, mo, subprocess, json):
    """GPU Detection and Validation"""
    
    def get_gpu_info() -> Dict:
        """Query NVIDIA GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                idx, name, mem, compute = line.split(', ')
                gpus.append({
                    'index': int(idx),
                    'name': name,
                    'memory_mb': int(mem),
                    'compute_capability': compute
                })
            return {'available': True, 'gpus': gpus}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    gpu_info = get_gpu_info()
    
    if gpu_info['available']:
        gpu_cards = mo.ui.table(
            data=gpu_info['gpus'],
            label="Available NVIDIA GPUs"
        )
        mo.callout(gpu_cards, kind="success")
    else:
        mo.callout(
            mo.md(f"⚠️ **No GPU detected**: {gpu_info.get('error', 'Unknown error')}"),
            kind="warn"
        )
    
    # Set PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return get_gpu_info, gpu_info, device


@app.cell
def __(mo, device, torch):
    """GPU Memory Monitor - Auto-refreshing"""
    
    def get_gpu_memory():
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': total - reserved,
                'total': total
            }
        return None
    
    # Auto-refresh every 2 seconds
    gpu_memory = mo.ui.refresh(
        lambda: get_gpu_memory(),
        options=[2, 5, 10],
        default_interval=2,
        label="GPU Memory (GB)"
    )
    
    return get_gpu_memory, gpu_memory


@app.cell
def __(mo, gpu_memory, px):
    """Display GPU Memory Bar Chart"""
    if gpu_memory.value:
        mem_data = gpu_memory.value
        fig = px.bar(
            x=['Allocated', 'Reserved', 'Free'],
            y=[mem_data['allocated'], mem_data['reserved'], mem_data['free']],
            title="GPU Memory Usage",
            labels={'x': '', 'y': 'GB'},
            color=['Allocated', 'Reserved', 'Free'],
            color_discrete_map={
                'Allocated': '#ff6b6b',
                'Reserved': '#ffa500', 
                'Free': '#51cf66'
            }
        )
        fig.update_layout(showlegend=False, height=300)
        mo.ui.plotly(fig)
    return


@app.cell
def __(mo, device, batch_size, precision, enable_cudnn_benchmark, torch):
    """Main Computation Cell - Replace with your GPU workload"""
    
    mo.md("## GPU Computation")
    
    # Configure PyTorch
    if enable_cudnn_benchmark.value:
        torch.backends.cudnn.benchmark = True
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    compute_dtype = dtype_map[precision.value]
    
    # Example: Random matrix multiplication
    size = batch_size.value * 128
    
    with torch.cuda.amp.autocast(enabled=(compute_dtype != torch.float32)):
        A = torch.randn(size, size, device=device, dtype=compute_dtype)
        B = torch.randn(size, size, device=device, dtype=compute_dtype)
        
        # Benchmark
        import time
        torch.cuda.synchronize()
        start = time.time()
        
        C = torch.matmul(A, B)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        gflops = (2 * size ** 3) / (elapsed * 1e9)
    
    result_stats = {
        'Matrix Size': f"{size}x{size}",
        'Precision': precision.value,
        'Time (ms)': f"{elapsed * 1000:.2f}",
        'GFLOPS': f"{gflops:.2f}",
        'GPU Memory (GB)': f"{torch.cuda.memory_allocated() / 1024**3:.2f}"
    }
    
    mo.ui.table(result_stats, label="Computation Results")
    return


@app.cell
def __(mo):
    """Results Visualization"""
    mo.md(
        """
        ## Results & Analysis
        
        [Add your domain-specific visualizations and insights here]
        """
    )
    return


@app.cell
def __(mo):
    """Export & Deployment Section"""
    mo.md(
        """
        ---
        
        ### 💾 Export Options
        
        This notebook can be:
        - **Run as a script**: `python nvidia_template.py`
        - **Deployed as an app**: `marimo run nvidia_template.py`
        - **Integrated in pipelines**: Import functions from this notebook
        
        ### 📚 Resources
        - [NVIDIA Documentation](https://docs.nvidia.com)
        - [Marimo Documentation](https://docs.marimo.io)
        - [Brev.dev Platform](https://brev.dev)
        """
    )
    return


if __name__ == "__main__":
    app.run()