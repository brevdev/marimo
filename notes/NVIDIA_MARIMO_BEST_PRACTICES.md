# NVIDIA + Marimo Notebook Best Practices

**Version**: 1.0  
**Date**: October 22, 2025  
**Purpose**: Combined best practices for creating production-ready NVIDIA GPU notebooks with Marimo

---

## Table of Contents

1. [Documentation & Structure](#1-documentation--structure)
2. [Code Organization](#2-code-organization)
3. [GPU Resource Management](#3-gpu-resource-management)
4. [Reactivity & Interactivity](#4-reactivity--interactivity)
5. [Performance & Optimization](#5-performance--optimization)
6. [Error Handling](#6-error-handling)
7. [Reproducibility](#7-reproducibility)
8. [User Experience](#8-user-experience)
9. [Testing & Validation](#9-testing--validation)
10. [Educational Value](#10-educational-value)

---

## 1. Documentation & Structure

### 1.1 Module Docstring ✅

**Requirements**:
- Clear title describing the notebook's purpose
- Brief description (2-3 sentences)
- Feature list (bullet points)
- GPU requirements (VRAM, compute capability)
- CUDA/driver version requirements
- Tested hardware configurations
- Author and date

**Example**:
```python
"""stable_diffusion_trt.py

Text-to-Image Generation with TensorRT
=======================================

Interactive Stable Diffusion implementation optimized with TensorRT for
fast inference on NVIDIA GPUs. Generate high-quality images from text prompts
with real-time parameter tuning.

Features:
- Text-to-image generation with adjustable parameters
- TensorRT optimization for 2-3x speedup
- Multiple image sizes (512x512, 768x768, etc.)
- Batch generation support
- GPU memory monitoring

Requirements:
- NVIDIA GPU with 8GB+ VRAM (any modern data center GPU)
- Works on: L40S (48GB), A100 (40/80GB), H100 (80GB), H200 (141GB), B200 (180GB), RTX PRO 6000 (48GB)
- CUDA 11.4+
- Stable Diffusion models (~4GB download on first run)
- Automatically adjusts settings based on available GPU memory
- Single GPU only (uses GPU 0)

Author: Brev.dev Team
Date: 2025-10-22
"""
```

### 1.2 Cell Organization 📦

**Best Practice**:
1. **Cell 1**: Imports and dependencies
2. **Cell 2**: Title and overview (markdown)
3. **Cell 3**: GPU detection and validation
4. **Cell 4+**: Configuration UI elements
5. **Middle cells**: Core logic and computation
6. **Final cells**: Results visualization and export

**Marimo Principle**: One logical responsibility per cell

**Example**:
```python
@app.cell
def __():
    """Import dependencies with automatic package management"""
    import marimo as mo
    import torch
    import numpy as np
    # ... other imports
    return mo, torch, np, ...

@app.cell
def __(mo):
    """Title and overview"""
    mo.md("""
    # NVIDIA GPU Demo: Title
    
    Description of what this notebook does...
    """)
    return

@app.cell
def __(mo, torch):
    """GPU detection and validation"""
    if not torch.cuda.is_available():
        mo.stop(
            mo.md("⚠️ **No GPU detected!** This notebook requires an NVIDIA GPU.").callout(kind="warn")
        )
    
    gpu_name = torch.cuda.get_device_properties(0).name
    return gpu_name,
```

### 1.3 Comments & Docstrings 📝

**Requirements**:
- Every function has a docstring
- Complex operations have inline comments
- Explain WHY, not just WHAT
- Reference NVIDIA documentation where relevant

**Example**:
```python
def optimize_with_tensorrt(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    precision: str = "fp16"
) -> Any:
    """
    Optimize PyTorch model with TensorRT for inference.
    
    TensorRT performs layer fusion, kernel auto-tuning, and precision
    calibration to dramatically improve inference performance on NVIDIA GPUs.
    
    Args:
        model: PyTorch model to optimize
        input_shape: Input tensor shape (batch, channels, height, width)
        precision: "fp32", "fp16", or "int8"
    
    Returns:
        TensorRT engine optimized for the target GPU
    
    References:
        - TensorRT Best Practices: https://docs.nvidia.com/deeplearning/tensorrt/
    """
    # Enable TensorRT optimizations
    # ... implementation
```

---

## 2. Code Organization

### 2.1 Minimal Global Variables 🌍

**Marimo Principle**: Keep global namespace clean

**Do ✅**:
```python
@app.cell
def __(data, model):
    # Use function-scoped variables
    _intermediate = process_data(data)  # Prefix with _ for local
    _result = model(_intermediate)
    
    final_output = _result.mean()  # Only return what's needed
    return final_output,
```

**Don't ❌**:
```python
@app.cell
def __(data, model):
    # Too many globals pollute namespace
    intermediate_step_1 = data.transform()
    intermediate_step_2 = intermediate_step_1.normalize()
    intermediate_step_3 = intermediate_step_2.scale()
    result_temp = model(intermediate_step_3)
    result_final = result_temp.mean()
    return intermediate_step_1, intermediate_step_2, intermediate_step_3, result_temp, result_final,
```

### 2.2 Descriptive Variable Names 📛

**Requirements**:
- Use descriptive names for global variables
- Prefix temporary/intermediate variables with `_`
- Follow PEP 8 naming conventions

**Examples**:
```python
# Good ✅
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
training_loss_history = []
model_checkpoint_path = "checkpoints/model.pth"

# Bad ❌
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
losses = []
path = "checkpoints/model.pth"
```

### 2.3 Function Encapsulation 🔨

**Marimo Principle**: Encapsulate logic to avoid namespace pollution

**Do ✅**:
```python
@app.cell
def __(torch, data):
    def train_model(data, epochs: int, lr: float):
        """Encapsulate training logic"""
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        losses = []
        for epoch in range(epochs):
            loss = train_epoch(model, data, optimizer)
            losses.append(loss)
        
        return model, losses
    
    # Only expose what's needed
    trained_model, loss_history = train_model(data, epochs=10, lr=0.001)
    return trained_model, loss_history
```

**Don't ❌**:
```python
@app.cell
def __(torch, data):
    # Pollutes namespace with intermediate variables
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    
    for epoch in range(10):
        for batch in data:
            optimizer.zero_grad()
            output = model(batch)
            loss = compute_loss(output, batch.labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    return model, optimizer, losses, epoch, batch, output, loss  # Too many!
```

### 2.4 Type Hints 🏷️

**Requirements**:
- All function parameters have type hints
- Return types specified
- Use `typing` module for complex types

**Example**:
```python
from typing import Dict, List, Optional, Tuple

def benchmark_gpu_operations(
    data_size: int,
    num_iterations: int,
    precision: str = "fp32"
) -> Dict[str, float]:
    """
    Benchmark GPU operations with specified configuration.
    
    Args:
        data_size: Number of elements to process
        num_iterations: Number of benchmark iterations
        precision: Computation precision ("fp32", "fp16", or "int8")
    
    Returns:
        Dictionary with timing metrics: {"mean": 1.23, "std": 0.05, "throughput": 1000}
    """
    # Implementation
    return {"mean": 1.23, "std": 0.05, "throughput": 1000.0}
```

---

## 3. GPU Resource Management

### 3.1 GPU Detection & Validation 🔍

**Requirements**:
- Always check GPU availability
- Display GPU information to user
- Graceful fallback or clear error message

**Example**:
```python
@app.cell
def __(mo, torch, subprocess):
    """GPU Detection and Information"""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        mo.stop(
            mo.callout(
                mo.md("""
                ⚠️ **No NVIDIA GPU Detected**
                
                This notebook requires an NVIDIA GPU with CUDA support.
                
                **Troubleshooting**:
                - Verify GPU is installed: `nvidia-smi`
                - Check CUDA driver version
                - Ensure PyTorch with CUDA support is installed
                """),
                kind="danger"
            )
        )
    
    # Get GPU information
    device = torch.device("cuda:0")
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = gpu_props.name
    gpu_memory_gb = gpu_props.total_memory / 1e9
    compute_capability = f"{gpu_props.major}.{gpu_props.minor}"
    
    # Display GPU info
    gpu_info = mo.md(f"""
    ✅ **GPU Detected**: {gpu_name}  
    💾 **Memory**: {gpu_memory_gb:.1f} GB  
    🔢 **Compute Capability**: {compute_capability}
    """).callout(kind="success")
    
    return device, gpu_name, gpu_memory_gb, gpu_info
```

### 3.2 Memory Monitoring 📊

**Requirements**:
- Real-time GPU memory usage display
- Use `mo.ui.refresh()` for auto-updating metrics
- Show both allocated and reserved memory

**Example**:
```python
@app.cell
def __(mo, torch):
    """Real-time GPU Memory Monitoring"""
    
    # Auto-refresh every 2 seconds
    refresh = mo.ui.refresh(default_interval="2s")
    
    def get_gpu_memory():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        utilization = (allocated / total) * 100
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization_pct": utilization
        }
    
    refresh
    return refresh, get_gpu_memory
```

### 3.3 Memory Cleanup 🧹

**Requirements**:
- Explicit cleanup of large tensors
- Use `torch.cuda.empty_cache()` after big operations
- Clear memory before expensive operations

**Example**:
```python
@app.cell
def __(torch, model, data):
    """Training with proper cleanup"""
    
    # Train model
    results = train_model(model, data)
    
    # Explicit cleanup
    del model  # Delete Python reference
    torch.cuda.empty_cache()  # Free cached memory
    
    return results,  # Don't return model if not needed
```

### 3.4 Memory-Aware Operations 🎯

**Requirements**:
- Scale operations based on available VRAM
- Provide automatic batch size adjustment
- Warn users about memory-intensive operations

**Example**:
```python
@app.cell
def __(mo, torch):
    """Memory-aware batch size selection"""
    
    # Detect available memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Adjust batch size based on GPU memory
    if gpu_memory_gb < 12:
        default_batch = 16
        max_batch = 32
        memory_warning = mo.callout(
            mo.md("⚠️ Limited GPU memory detected. Reduced batch size for stability."),
            kind="warn"
        )
    elif gpu_memory_gb < 24:
        default_batch = 32
        max_batch = 64
        memory_warning = None
    else:
        default_batch = 64
        max_batch = 128
        memory_warning = None
    
    batch_size = mo.ui.slider(
        start=8,
        stop=max_batch,
        step=8,
        value=default_batch,
        label="Batch Size",
        show_value=True
    )
    
    return batch_size, memory_warning
```

---

## 4. Reactivity & Interactivity

### 4.1 Reactive Execution ⚛️

**Marimo Principle**: Use reactive execution, NOT callbacks

**Do ✅**:
```python
@app.cell
def __(mo):
    """Interactive controls"""
    learning_rate = mo.ui.slider(0.0001, 0.01, value=0.001, label="Learning Rate")
    batch_size = mo.ui.slider(16, 128, step=16, value=32, label="Batch Size")
    return learning_rate, batch_size

@app.cell
def __(learning_rate, batch_size):
    """This cell automatically re-runs when learning_rate or batch_size change"""
    config = {
        "lr": learning_rate.value,
        "batch": batch_size.value
    }
    return config,
```

**Don't ❌**:
```python
@app.cell
def __(mo):
    """DON'T use on_change handlers"""
    def handle_change(value):
        # This breaks Marimo's reactive paradigm
        update_model(value)
    
    slider = mo.ui.slider(0, 100, on_change=handle_change)  # ❌ Don't do this
    return slider,
```

### 4.2 UI Element Returns 📤

**Requirements**:
- All UI elements must be returned from their cell
- Reference `.value` in dependent cells
- Use descriptive variable names

**Example**:
```python
@app.cell
def __(mo):
    """Configuration UI"""
    num_epochs = mo.ui.slider(1, 100, value=10, label="Training Epochs")
    learning_rate = mo.ui.number(0.001, label="Learning Rate", step=0.0001)
    optimizer_type = mo.ui.dropdown(
        ["Adam", "SGD", "AdamW"],
        value="Adam",
        label="Optimizer"
    )
    
    # Display UI
    config_ui = mo.vstack([
        mo.md("## Training Configuration"),
        num_epochs,
        learning_rate,
        optimizer_type
    ])
    
    config_ui
    return num_epochs, learning_rate, optimizer_type
```

### 4.3 Expensive Operations with mo.stop() 💰

**Marimo Principle**: Use `mo.stop()` to prevent expensive operations until ready

**Example**:
```python
@app.cell
def __(mo):
    """Run button for expensive training"""
    train_button = mo.ui.run_button(label="🚀 Start Training")
    return train_button,

@app.cell
def __(mo, train_button, model, data):
    """Training only runs when button clicked"""
    
    # Stop execution until button is clicked
    mo.stop(not train_button.value, mo.md("Click the button to start training"))
    
    # Expensive operation
    with mo.status.spinner(title="Training model...", subtitle="This may take a few minutes"):
        results = train_model(model, data)
    
    return results,
```

### 4.4 Progress Indicators 🔄

**Requirements**:
- Use `mo.status.spinner()` for long operations
- Show progress messages
- Indicate time estimates

**Example**:
```python
@app.cell
def __(mo, subprocess):
    """Installation with progress indicator"""
    
    with mo.status.spinner(
        title="📦 Installing CUDA toolkit...",
        subtitle="Downloading ~2GB of packages (2-3 minutes)"
    ):
        result = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"],
            capture_output=True,
            timeout=300
        )
    
    if result.returncode == 0:
        message = mo.md("✅ Installation complete!").callout(kind="success")
    else:
        message = mo.md("❌ Installation failed").callout(kind="danger")
    
    return message,
```

---

## 5. Performance & Optimization

### 5.1 GPU Utilization Metrics ⚡

**Requirements**:
- Display real-time GPU utilization
- Show throughput metrics
- Compare CPU vs GPU performance

**Example**:
```python
@app.cell
def __(mo, torch, time):
    """Performance benchmark with metrics"""
    
    def benchmark_operation(size: int, device: str):
        """Benchmark matrix multiplication"""
        # Create tensors
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        _ = torch.matmul(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate metrics
        flops = 2 * size**3 * 100  # Matrix multiply FLOPs
        throughput = flops / elapsed / 1e12  # TFLOPS
        
        return {
            "elapsed_sec": elapsed,
            "throughput_tflops": throughput,
            "speedup": None  # Calculated later
        }
    
    return benchmark_operation,
```

### 5.2 Mixed Precision ��️

**Requirements**:
- Use FP16/BF16 for suitable operations
- Document precision choices
- Show performance impact

**Example**:
```python
@app.cell
def __(torch, model):
    """Mixed precision training for 2x speedup"""
    
    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    def train_with_amp(model, data, optimizer):
        """Train with automatic mixed precision"""
        for batch in data:
            optimizer.zero_grad()
            
            # Forward pass in FP16
            with torch.cuda.amp.autocast():
                outputs = model(batch.inputs)
                loss = compute_loss(outputs, batch.labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        return loss.item()
    
    return train_with_amp,
```

### 5.3 Efficient Memory Transfers 🔄

**Requirements**:
- Minimize CPU↔GPU transfers
- Pin memory for faster transfers
- Batch operations when possible

**Example**:
```python
@app.cell
def __(torch):
    """Efficient data loading"""
    
    # Pin memory for faster transfers
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,  # Faster CPU→GPU transfer
        num_workers=4
    )
    
    # Keep data on GPU between operations
    def process_batch_efficiently(batch):
        # Transfer once
        batch_gpu = batch.to("cuda", non_blocking=True)
        
        # All operations on GPU
        result1 = operation1(batch_gpu)
        result2 = operation2(result1)
        result3 = operation3(result2)
        
        # Transfer back only final result
        return result3.cpu()
    
    return process_batch_efficiently,
```

### 5.4 Caching Expensive Computations 💾

**Marimo Principle**: Cache results of expensive operations

**Example**:
```python
@app.cell
def __(mo, torch):
    """Cache expensive model loading"""
    
    @mo.cache
    def load_model(model_name: str):
        """Load model with caching (only loads once)"""
        import time
        print(f"Loading {model_name}...")  # Only prints on first call
        time.sleep(2)  # Simulate slow loading
        model = create_model(model_name)
        return model
    
    # This will use cached result on subsequent runs
    model = load_model("large-model-v1")
    
    return model,
```

---

## 6. Error Handling

### 6.1 GPU Error Handling 🛡️

**Requirements**:
- Catch and handle GPU OOM errors
- Provide helpful error messages
- Suggest fixes to user

**Example**:
```python
@app.cell
def __(mo, torch):
    """GPU operation with error handling"""
    
    def safe_gpu_operation(size: int):
        """Perform operation with OOM protection"""
        try:
            # Attempt operation
            tensor = torch.randn(size, size, device="cuda")
            result = torch.matmul(tensor, tensor)
            return result
            
        except torch.cuda.OutOfMemoryError:
            # Clean up
            torch.cuda.empty_cache()
            
            # Helpful error message
            error_msg = mo.callout(
                mo.md(f"""
                ❌ **GPU Out of Memory**
                
                Tried to allocate {size}x{size} matrix ({size**2*4/1e9:.2f} GB)
                
                **Solutions**:
                - Reduce batch size or matrix dimensions
                - Close other GPU applications
                - Use a GPU with more memory
                
                **Current GPU**: {torch.cuda.get_device_properties(0).name}  
                **Available Memory**: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB
                """),
                kind="danger"
            )
            mo.stop(True, error_msg)
            
        except Exception as e:
            error_msg = mo.md(f"❌ **Error**: {str(e)}").callout(kind="danger")
            mo.stop(True, error_msg)
    
    return safe_gpu_operation,
```

### 6.2 Dependency Handling 📦

**Requirements**:
- Graceful handling of missing dependencies
- Clear installation instructions
- Optional features with fallbacks

**Example**:
```python
@app.cell
def __(mo):
    """Import with fallback"""
    
    # Try importing cuDF (GPU)
    try:
        import cudf
        has_cudf = True
        backend_msg = mo.md("✅ Using **cuDF** (GPU-accelerated)").callout(kind="success")
    except ImportError:
        import pandas as pd
        cudf = pd  # Fallback to pandas
        has_cudf = False
        backend_msg = mo.callout(
            mo.md("""
            ℹ️ **cuDF not available** - using Pandas (CPU)
            
            For GPU acceleration, install RAPIDS cuDF:
            ```bash
            conda install -c rapidsai -c nvidia cudf
            ```
            """),
            kind="info"
        )
    
    backend_msg
    return cudf, has_cudf
```

### 6.3 User Input Validation ✓

**Requirements**:
- Validate all user inputs
- Provide clear error messages
- Suggest valid ranges

**Example**:
```python
@app.cell
def __(mo, batch_size, learning_rate):
    """Validate configuration"""
    
    errors = []
    
    # Validate batch size
    if batch_size.value < 1:
        errors.append("Batch size must be at least 1")
    if batch_size.value > 512:
        errors.append("Batch size too large (max 512)")
    
    # Validate learning rate
    if learning_rate.value <= 0:
        errors.append("Learning rate must be positive")
    if learning_rate.value > 1.0:
        errors.append("Learning rate too large (typically < 0.1)")
    
    # Stop if errors
    if errors:
        error_display = mo.callout(
            mo.md("❌ **Invalid Configuration**\n\n" + "\n".join(f"- {e}" for e in errors)),
            kind="danger"
        )
        mo.stop(True, error_display)
    
    return
```

---

## 7. Reproducibility

### 7.1 Random Seeds 🎲

**Requirements**:
- Set seeds for reproducibility
- Document seed values
- Allow user control

**Example**:
```python
@app.cell
def __(torch, np, random):
    """Set random seeds for reproducibility"""
    
    def set_seed(seed: int = 42):
        """Set all random seeds for reproducible results"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set default seed
    set_seed(42)
    
    return set_seed,
```

### 7.2 Environment Documentation 📋

**Requirements**:
- Document all version requirements
- Test on specified configurations
- Note any platform-specific behavior

**Example** (in docstring):
```python
"""
Requirements:
- Python 3.8+
- PyTorch 2.0+ with CUDA 11.8+
- Transformers 4.30+
- CUDA Toolkit 11.8+ (for gpu-burn compilation)
- NVIDIA Driver 520+

Tested Configurations:
- Ubuntu 22.04 + L4 (23GB) + CUDA 12.8 ✅
- Ubuntu 22.04 + A100 (40GB) + CUDA 12.1 ✅  
- Ubuntu 20.04 + A100 (80GB) + CUDA 11.8 ✅

Known Issues:
- TensorRT optimization requires compute capability 7.0+ (Volta or newer)
- Batch size >64 may cause OOM on GPUs with <16GB VRAM
"""
```

### 7.3 Idempotent Cells 🔁

**Marimo Principle**: Cells should produce same output given same inputs

**Do ✅**:
```python
@app.cell
def __(data, seed):
    """Idempotent: same data + seed → same result"""
    torch.manual_seed(seed)
    shuffled = data[torch.randperm(len(data))]
    return shuffled,
```

**Don't ❌**:
```python
@app.cell
def __(data):
    """Non-idempotent: different result each run"""
    # No seed = different results each time
    shuffled = data[torch.randperm(len(data))]
    return shuffled,
```

---

## 8. User Experience

### 8.1 Visual Layout 🎨

**Requirements**:
- Use `mo.hstack()` and `mo.vstack()` for clean layouts
- Group related UI elements
- Use callouts for important messages
- Consistent styling

**Example**:
```python
@app.cell
def __(mo, learning_rate, batch_size, num_epochs):
    """Clean configuration layout"""
    
    config_panel = mo.vstack([
        mo.md("## Training Configuration"),
        mo.hstack([
            mo.vstack([
                mo.md("**Learning Rate**"),
                learning_rate
            ]),
            mo.vstack([
                mo.md("**Batch Size**"),
                batch_size
            ]),
            mo.vstack([
                mo.md("**Epochs**"),
                num_epochs
            ])
        ]),
        mo.callout(
            mo.md("💡 **Tip**: Start with default values, then tune based on results"),
            kind="info"
        )
    ])
    
    config_panel
    return config_panel,
```

### 8.2 Informative Feedback 💬

**Requirements**:
- Show progress during long operations
- Display results clearly
- Provide context and interpretation

**Example**:
```python
@app.cell
def __(mo, training_results):
    """Display results with context"""
    
    results_display = mo.vstack([
        mo.md("## Training Results"),
        
        mo.callout(
            mo.md(f"""
            ✅ **Training Complete!**
            
            - Final Loss: {training_results['loss']:.4f}
            - Accuracy: {training_results['accuracy']:.2%}
            - Training Time: {training_results['time']:.1f}s
            - GPU Utilization: {training_results['gpu_util']:.1f}%
            """),
            kind="success"
        ),
        
        mo.md("### Loss Curve"),
        plot_loss_curve(training_results['loss_history']),
        
        mo.callout(
            mo.md("""
            💡 **Interpretation**:
            - Loss is decreasing steadily ✅
            - No signs of overfitting
            - Consider training for more epochs
            """),
            kind="info"
        )
    ])
    
    results_display
    return results_display,
```

### 8.3 Interactive Visualizations 📊

**Requirements**:
- Use Plotly for interactive charts
- Enable zoom, pan, hover tooltips
- Show multiple perspectives

**Example**:
```python
@app.cell
def __(mo, go, training_history):
    """Interactive training metrics"""
    
    fig = go.Figure()
    
    # Add loss trace
    fig.add_trace(go.Scatter(
        x=list(range(len(training_history['loss']))),
        y=training_history['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='red', width=2)
    ))
    
    # Add accuracy trace
    fig.add_trace(go.Scatter(
        x=list(range(len(training_history['accuracy']))),
        y=training_history['accuracy'],
        mode='lines',
        name='Accuracy',
        line=dict(color='green', width=2),
        yaxis='y2'
    ))
    
    # Layout with dual y-axes
    fig.update_layout(
        title='Training Metrics',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss', side='left'),
        yaxis2=dict(title='Accuracy', side='right', overlaying='y'),
        hovermode='x unified',
        height=400
    )
    
    mo.ui.plotly(fig)
    return
```

---

## 9. Testing & Validation

### 9.1 Self-Test Cells 🧪

**Requirements**:
- Include sanity checks
- Validate GPU operations
- Test edge cases

**Example**:
```python
@app.cell
def __(mo, torch):
    """GPU capability tests"""
    
    tests_passed = []
    tests_failed = []
    
    # Test 1: Basic tensor operations
    try:
        a = torch.randn(100, 100, device='cuda')
        b = torch.matmul(a, a)
        assert b.shape == (100, 100)
        tests_passed.append("✅ Basic tensor operations")
    except Exception as e:
        tests_failed.append(f"❌ Basic operations: {e}")
    
    # Test 2: Memory allocation
    try:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        large = torch.randn(1000, 1000, device='cuda')
        del large
        torch.cuda.empty_cache()
        tests_passed.append(f"✅ Memory operations ({mem_gb:.1f} GB available)")
    except Exception as e:
        tests_failed.append(f"❌ Memory operations: {e}")
    
    # Display results
    if tests_failed:
        test_results = mo.callout(
            mo.md("**Tests**\n\n" + "\n".join(tests_passed + tests_failed)),
            kind="warn"
        )
    else:
        test_results = mo.callout(
            mo.md("**All Tests Passed** ✅\n\n" + "\n".join(tests_passed)),
            kind="success"
        )
    
    test_results
    return test_results,
```

### 9.2 Performance Benchmarks 📈

**Requirements**:
- Include reproducible benchmarks
- Show GPU vs CPU comparison
- Document expected performance

**Example**:
```python
@app.cell
def __(mo, benchmark_operation):
    """Performance comparison"""
    
    size = 4096
    
    # Run benchmarks
    cpu_results = benchmark_operation(size, "cpu")
    gpu_results = benchmark_operation(size, "cuda")
    
    speedup = cpu_results['elapsed_sec'] / gpu_results['elapsed_sec']
    
    # Display comparison
    comparison = mo.md(f"""
    ## Performance Comparison
    
    **Matrix Size**: {size}x{size}
    
    | Device | Time (s) | Throughput (TFLOPS) | Speedup |
    |--------|----------|---------------------|---------|
    | CPU | {cpu_results['elapsed_sec']:.3f} | {cpu_results['throughput_tflops']:.2f} | 1.0x |
    | GPU | {gpu_results['elapsed_sec']:.3f} | {gpu_results['throughput_tflops']:.2f} | **{speedup:.1f}x** |
    
    🚀 GPU is **{speedup:.1f}x faster** than CPU!
    """).callout(kind="success" if speedup > 5 else "info")
    
    comparison
    return speedup, comparison
```

---

## 10. Educational Value

### 10.1 Explain Concepts 📚

**Requirements**:
- Explain WHY choices are made
- Link to documentation
- Provide learning resources

**Example**:
```python
@app.cell
def __(mo):
    """Explain mixed precision training"""
    
    mo.md("""
    ## Mixed Precision Training 🎯
    
    **What is it?**  
    Using FP16 (half precision) for most operations while keeping FP32 for 
    critical operations like loss scaling.
    
    **Why use it?**
    - **2x faster** training on modern NVIDIA GPUs (Volta+)
    - **50% less memory** usage → larger batch sizes
    - **Minimal accuracy impact** with proper loss scaling
    
    **How it works:**
    1. Forward pass in FP16 (fast matrix math)
    2. Loss scaling prevents underflow
    3. Backward pass with gradient scaling
    4. Optimizer updates in FP32 (accuracy)
    
    **Learn more:**
    - [NVIDIA Mixed Precision Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
    - [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
    """)
    return
```

### 10.2 Show Best Practices in Action 💡

**Requirements**:
- Demonstrate optimal patterns
- Explain common pitfalls
- Provide before/after examples

**Example**:
```python
@app.cell
def __(mo):
    """Demonstrate GPU memory best practices"""
    
    mo.md("""
    ## GPU Memory Best Practices 💾
    
    ### ❌ Bad: Unnecessary transfers
    ```python
    for batch in data:
        batch_gpu = batch.to('cuda')  # Transfer on every iteration
        result = model(batch_gpu)
        result_cpu = result.cpu()     # Transfer back
        process(result_cpu)
    ```
    
    ### ✅ Good: Batch transfers
    ```python
    model.cuda()  # Move model once
    for batch in data:
        batch_gpu = batch.to('cuda')   # Transfer input
        result = model(batch_gpu)      # Compute on GPU
        # Keep intermediate results on GPU
    final_result = collect_results().cpu()  # Transfer once at end
    ```
    
    **Why this matters:**
    - CPU↔GPU transfers are SLOW (PCIe bandwidth limited)
    - Keep data on GPU between operations
    - Transfer only when necessary
    
    **Rule of thumb:** Minimize transfers, maximize GPU compute
    """)
    return
```

### 10.3 Interactive Learning 🎓

**Requirements**:
- Let users experiment
- Show immediate feedback
- Encourage exploration

**Example**:
```python
@app.cell
def __(mo):
    """Interactive parameter exploration"""
    
    mo.md("""
    ## Experiment: Batch Size Impact 🧪
    
    **Try different batch sizes** and observe:
    - Training speed
    - GPU memory usage
    - Model convergence
    
    **Hypothesis:** Larger batches = faster training but more memory
    
    **Your task:** Find the optimal batch size for this GPU!
    """)
    
    batch_size_experiment = mo.ui.slider(8, 128, step=8, value=32, label="Batch Size")
    
    mo.vstack([
        batch_size_experiment,
        mo.md("👆 Adjust the slider and click 'Run Training' to see the effect")
    ])
    return batch_size_experiment,
```

---

## Validation Checklist ✅

Use this checklist to validate notebooks:

### Documentation
- [ ] Module docstring with title, description, features
- [ ] GPU requirements clearly stated
- [ ] Tested hardware configurations listed
- [ ] Cell-level comments for complex operations
- [ ] Function docstrings with type hints

### Code Organization
- [ ] Minimal global variables (< 10 per notebook)
- [ ] Descriptive variable names
- [ ] Logic encapsulated in functions
- [ ] Proper cell organization (imports → config → logic → viz)
- [ ] Type hints on all functions

### GPU Resource Management
- [ ] GPU detection and validation
- [ ] Real-time memory monitoring
- [ ] Proper cleanup (`del`, `empty_cache()`)
- [ ] Memory-aware operations
- [ ] Graceful handling of GPU errors

### Reactivity & Interactivity
- [ ] All UI elements returned from cells
- [ ] Uses reactive execution (no `on_change`)
- [ ] `mo.stop()` for expensive operations
- [ ] Progress indicators for long operations
- [ ] Proper use of `mo.ui.run_button()`

### Performance
- [ ] GPU utilization metrics displayed
- [ ] CPU vs GPU comparison
- [ ] Mixed precision where appropriate
- [ ] Efficient memory transfers
- [ ] Caching of expensive operations

### Error Handling
- [ ] GPU OOM errors caught and handled
- [ ] Missing dependencies handled gracefully
- [ ] User input validation
- [ ] Clear error messages with solutions
- [ ] Fallback options where possible

### Reproducibility
- [ ] Random seeds set and documented
- [ ] Deterministic operations
- [ ] Environment documented
- [ ] Idempotent cells
- [ ] Version requirements specified

### User Experience
- [ ] Clean visual layout
- [ ] Informative feedback
- [ ] Interactive visualizations
- [ ] Callouts for important info
- [ ] Consistent styling

### Testing & Validation
- [ ] Self-test cells included
- [ ] Performance benchmarks
- [ ] Edge cases tested
- [ ] Results validated
- [ ] Examples work end-to-end

### Educational Value
- [ ] Concepts explained clearly
- [ ] Links to documentation
- [ ] Best practices demonstrated
- [ ] Common pitfalls addressed
- [ ] Interactive learning elements

---

## Common Pitfalls to Avoid ⚠️

### 1. Breaking Marimo Reactivity
❌ Using `on_change` callbacks  
✅ Use reactive cell dependencies

### 2. GPU Memory Leaks
❌ Not deleting large tensors  
✅ Explicit `del` + `empty_cache()`

### 3. Excessive Global Variables
❌ Returning all intermediate variables  
✅ Prefix temps with `_` or use functions

### 4. Poor Error Messages
❌ "Error occurred"  
✅ "GPU OOM: Reduce batch size from 64 to 32"

### 5. No Progress Indicators
❌ UI freezes during long operations  
✅ Use `mo.status.spinner()`

### 6. Ignoring Edge Cases
❌ Assume 40GB GPU  
✅ Adapt to available memory

### 7. Missing Documentation
❌ No comments on complex operations  
✅ Explain WHY, not just WHAT

### 8. Non-Reproducible Results
❌ No random seeds  
✅ Set seeds + document them

### 9. Poor Visual Layout
❌ Everything in one column  
✅ Use `hstack`/`vstack` for organization

### 10. Not Educational
❌ Just show code  
✅ Explain concepts + link to resources

---

## Quick Reference Card 📇

```python
# GPU Detection
if not torch.cuda.is_available():
    mo.stop(True, mo.md("⚠️ No GPU").callout(kind="warn"))

# Memory Monitoring
allocated = torch.cuda.memory_allocated(0) / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9

# Progress Indicator
with mo.status.spinner(title="Processing...", subtitle="Please wait"):
    result = expensive_operation()

# Conditional Execution
mo.stop(not button.value, mo.md("Click button to start"))

# UI Elements
slider = mo.ui.slider(0, 100, value=50, label="Parameter")
button = mo.ui.run_button(label="Start")

# Layout
mo.vstack([header, content, footer])
mo.hstack([left_panel, right_panel])

# Callouts
mo.callout(mo.md("**Important info**"), kind="info")  # info, warn, danger, success

# Error Handling
try:
    result = gpu_operation()
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    mo.stop(True, mo.md("❌ OOM").callout(kind="danger"))

# Caching
@mo.cache
def expensive_function():
    return slow_computation()
```

---

## Resources 📚

### NVIDIA Documentation
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [RAPIDS Documentation](https://docs.rapids.ai/)

### Marimo Documentation
- [Best Practices](https://docs.marimo.io/guides/best_practices/)
- [Reactive Execution](https://docs.marimo.io/guides/reactivity/)
- [UI Elements](https://docs.marimo.io/api/inputs/)
- [Expensive Notebooks](https://docs.marimo.io/guides/expensive_notebooks/)

---

**Version History**:
- v1.0 (2025-10-22): Initial combined best practices guide

**Maintained by**: Brev.dev Team  
**Last Updated**: October 22, 2025

