"""tensorrt_optimization.py

TensorRT Model Optimization Pipeline
=====================================

Interactive pipeline for optimizing deep learning models with NVIDIA TensorRT.
Compare original PyTorch models with TensorRT-optimized versions, showing
dramatic improvements in inference latency and throughput.

Features:
- Convert PyTorch models to TensorRT engines
- Compare FP32, FP16, and INT8 precision modes
- Real-time latency and throughput benchmarking
- Interactive batch size tuning
- Memory usage comparison
- Support for various model architectures

Requirements:
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA 11.4+
- TensorRT 8.5+
- 8GB+ GPU memory

Author: Brev.dev Team
Date: 2025-10-17
"""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def __():
    """Import dependencies"""
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    import plotly.graph_objects as go
    from typing import Dict, List, Optional, Tuple
    import subprocess
    import time
    
    # Try importing TensorRT
    try:
        import tensorrt as trt
        import torch_tensorrt
        TRT_AVAILABLE = True
    except ImportError:
        TRT_AVAILABLE = False
        trt = None
        torch_tensorrt = None
    
    return (
        mo, torch, nn, np, go, Dict, List, Optional, Tuple,
        subprocess, time, trt, torch_tensorrt, TRT_AVAILABLE
    )


@app.cell
def __(mo, TRT_AVAILABLE):
    """Title and TensorRT availability"""
    mo.md(
        f"""
        # ⚡ TensorRT Model Optimization
        
        **Accelerate inference with NVIDIA TensorRT** - the SDK for high-performance
        deep learning inference. TensorRT provides INT8 and FP16 optimizations,
        layer fusion, kernel tuning, and dynamic tensor memory management.
        
        **TensorRT Status**: {'✅ Available' if TRT_AVAILABLE else '⚠️ Not installed'}
        
        ## ⚙️ Optimization Configuration
        """
    )
    return


@app.cell
def __(mo):
    """Interactive optimization controls"""
    model_choice = mo.ui.dropdown(
        options=['ResNet18', 'ResNet50', 'MobileNetV2', 'EfficientNet-B0'],
        value='ResNet18',
        label="Model Architecture"
    )
    
    precision_mode = mo.ui.dropdown(
        options=['FP32', 'FP16', 'INT8'],
        value='FP16',
        label="Precision Mode"
    )
    
    batch_size = mo.ui.slider(
        start=1, stop=64, step=1, value=8,
        label="Batch Size", show_value=True
    )
    
    num_iterations = mo.ui.slider(
        start=10, stop=500, step=10, value=100,
        label="Benchmark Iterations", show_value=True
    )
    
    optimize_btn = mo.ui.run_button(label="🚀 Optimize & Benchmark")
    
    mo.vstack([
        mo.hstack([model_choice, precision_mode], justify="start"),
        mo.hstack([batch_size, num_iterations], justify="start"),
        optimize_btn
    ])
    return model_choice, precision_mode, batch_size, num_iterations, optimize_btn


@app.cell
def __(torch, mo, subprocess):
    """GPU Detection"""
    
    def get_gpu_info() -> Dict:
        """Query NVIDIA GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    idx, name, mem, compute = line.split(', ')
                    gpus.append({
                        'GPU': int(idx),
                        'Model': name,
                        'Memory (GB)': f"{int(mem) / 1024:.1f}",
                        'Compute Cap': compute
                    })
            return {'available': True, 'gpus': gpus}
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    gpu_info = get_gpu_info()
    
    if gpu_info['available']:
        mo.callout(
            mo.vstack([
                mo.md("**✅ GPU Ready for TensorRT**"),
                mo.ui.table(gpu_info['gpus'])
            ]),
            kind="success"
        )
    else:
        mo.callout(
            mo.md(f"⚠️ **No GPU detected**: {gpu_info.get('error', 'Unknown')}"),
            kind="warn"
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return get_gpu_info, gpu_info, device


@app.cell
def __(mo, device, torch):
    """GPU Memory Monitor"""
    
    def get_gpu_memory() -> Optional[Dict]:
        """Get current GPU memory usage"""
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'Allocated': f"{allocated:.2f} GB",
                'Reserved': f"{reserved:.2f} GB",
                'Free': f"{total - reserved:.2f} GB"
            }
        return None
    
    gpu_memory = mo.ui.refresh(
        lambda: get_gpu_memory(),
        options=[2, 5, 10],
        default_interval=3
    )
    
    mo.vstack([
        mo.md("### 📊 GPU Memory"),
        gpu_memory if gpu_memory.value else mo.md("*CPU mode*")
    ])
    return get_gpu_memory, gpu_memory


@app.cell
def __(torch, nn):
    """Model definitions"""
    
    def get_model(model_name: str, pretrained: bool = False):
        """Get model by name"""
        if model_name == 'ResNet18':
            from torchvision.models import resnet18, ResNet18_Weights
            return resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        elif model_name == 'ResNet50':
            from torchvision.models import resnet50, ResNet50_Weights
            return resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name == 'MobileNetV2':
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        elif model_name == 'EfficientNet-B0':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    return get_model,


@app.cell
def __(torch, time, np):
    """Benchmarking utilities"""
    
    def benchmark_model(
        model, 
        input_shape: Tuple[int, ...], 
        num_iterations: int = 100,
        warmup: int = 10,
        device: str = 'cuda'
    ) -> Dict:
        """Benchmark model inference performance"""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(dummy_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                latencies.append(time.time() - start)
        
        # Calculate statistics
        latencies = np.array(latencies) * 1000  # Convert to ms
        
        return {
            'mean_latency': float(np.mean(latencies)),
            'std_latency': float(np.std(latencies)),
            'p50_latency': float(np.percentile(latencies, 50)),
            'p95_latency': float(np.percentile(latencies, 95)),
            'p99_latency': float(np.percentile(latencies, 99)),
            'throughput': 1000.0 / np.mean(latencies),  # images/sec
            'all_latencies': latencies.tolist()
        }
    
    return benchmark_model,


@app.cell
def __(
    optimize_btn, model_choice, precision_mode, batch_size,
    num_iterations, get_model, device, benchmark_model,
    torch, TRT_AVAILABLE, torch_tensorrt, mo
):
    """Optimization and benchmarking"""
    
    optimization_results = None
    
    if optimize_btn.value:
        mo.md("### 🔄 Optimizing model...")
        
        try:
            # Load model
            model = get_model(model_choice.value, pretrained=True)
            model = model.to(device).eval()
            
            input_shape = (batch_size.value, 3, 224, 224)
            
            # Benchmark original PyTorch model
            mo.md("Benchmarking PyTorch model...")
            pytorch_results = benchmark_model(
                model, 
                input_shape, 
                num_iterations=num_iterations.value,
                device=str(device)
            )
            
            # TensorRT optimization
            trt_results = None
            if TRT_AVAILABLE and device.type == 'cuda':
                try:
                    mo.md("Converting to TensorRT...")
                    
                    # Prepare example input
                    example_input = torch.randn(input_shape, device=device)
                    
                    # Set precision
                    enabled_precisions = {torch.float32}
                    if precision_mode.value == 'FP16':
                        enabled_precisions.add(torch.float16)
                    elif precision_mode.value == 'INT8':
                        enabled_precisions.add(torch.float16)
                        enabled_precisions.add(torch.int8)
                    
                    # Compile with Torch-TensorRT
                    trt_model = torch_tensorrt.compile(
                        model,
                        inputs=[torch_tensorrt.Input(input_shape)],
                        enabled_precisions=enabled_precisions,
                        workspace_size=1 << 30,  # 1GB
                    )
                    
                    mo.md("Benchmarking TensorRT model...")
                    trt_results = benchmark_model(
                        trt_model,
                        input_shape,
                        num_iterations=num_iterations.value,
                        device=str(device)
                    )
                    
                except Exception as trt_error:
                    trt_results = {'error': str(trt_error)}
            
            optimization_results = {
                'model_name': model_choice.value,
                'precision': precision_mode.value,
                'batch_size': batch_size.value,
                'input_shape': input_shape,
                'pytorch': pytorch_results,
                'tensorrt': trt_results,
                'success': True
            }
            
        except Exception as e:
            optimization_results = {
                'error': str(e),
                'success': False
            }
    
    return optimization_results,


@app.cell
def __(optimization_results, mo, go, TRT_AVAILABLE):
    """Visualize optimization results"""
    
    if optimization_results is None:
        mo.callout(
            mo.md("**Click 'Optimize & Benchmark' to start**"),
            kind="info"
        )
    elif not optimization_results.get('success', False):
        mo.callout(
            mo.md(f"**Error**: {optimization_results.get('error', 'Unknown error')}"),
            kind="danger"
        )
    else:
        pytorch_res = optimization_results['pytorch']
        trt_res = optimization_results['tensorrt']
        
        # Results table
        table_data = {
            'Metric': [
                'Mean Latency (ms)',
                'P50 Latency (ms)',
                'P95 Latency (ms)',
                'P99 Latency (ms)',
                'Throughput (img/s)',
                'Std Dev (ms)'
            ],
            'PyTorch': [
                f"{pytorch_res['mean_latency']:.2f}",
                f"{pytorch_res['p50_latency']:.2f}",
                f"{pytorch_res['p95_latency']:.2f}",
                f"{pytorch_res['p99_latency']:.2f}",
                f"{pytorch_res['throughput']:.1f}",
                f"{pytorch_res['std_latency']:.2f}"
            ]
        }
        
        if trt_res and 'error' not in trt_res:
            table_data['TensorRT'] = [
                f"{trt_res['mean_latency']:.2f}",
                f"{trt_res['p50_latency']:.2f}",
                f"{trt_res['p95_latency']:.2f}",
                f"{trt_res['p99_latency']:.2f}",
                f"{trt_res['throughput']:.1f}",
                f"{trt_res['std_latency']:.2f}"
            ]
            
            speedup = pytorch_res['mean_latency'] / trt_res['mean_latency']
            throughput_gain = trt_res['throughput'] / pytorch_res['throughput']
            
            table_data['Speedup'] = [
                f"{speedup:.2f}x",
                f"{pytorch_res['p50_latency'] / trt_res['p50_latency']:.2f}x",
                f"{pytorch_res['p95_latency'] / trt_res['p95_latency']:.2f}x",
                f"{pytorch_res['p99_latency'] / trt_res['p99_latency']:.2f}x",
                f"{throughput_gain:.2f}x",
                "-"
            ]
        
        # Latency distribution chart
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=pytorch_res['all_latencies'],
            name='PyTorch',
            marker_color='#ff6b6b'
        ))
        
        if trt_res and 'error' not in trt_res:
            fig.add_trace(go.Box(
                y=trt_res['all_latencies'],
                name='TensorRT',
                marker_color='#51cf66'
            ))
        
        fig.update_layout(
            title="Latency Distribution",
            yaxis_title="Latency (ms)",
            height=400,
            showlegend=True
        )
        
        # Speedup visualization
        if trt_res and 'error' not in trt_res:
            fig_speedup = go.Figure()
            
            metrics = ['Mean', 'P50', 'P95', 'P99']
            speedups = [
                pytorch_res['mean_latency'] / trt_res['mean_latency'],
                pytorch_res['p50_latency'] / trt_res['p50_latency'],
                pytorch_res['p95_latency'] / trt_res['p95_latency'],
                pytorch_res['p99_latency'] / trt_res['p99_latency']
            ]
            
            fig_speedup.add_trace(go.Bar(
                x=metrics,
                y=speedups,
                marker_color='#4c6ef5',
                text=[f"{s:.2f}x" for s in speedups],
                textposition='outside'
            ))
            
            fig_speedup.update_layout(
                title="TensorRT Speedup",
                xaxis_title="Latency Metric",
                yaxis_title="Speedup Factor",
                height=350
            )
            
            mo.vstack([
                mo.md(f"### ✅ Optimization Complete!"),
                mo.md(f"**Model**: {optimization_results['model_name']} | **Precision**: {optimization_results['precision']} | **Batch Size**: {optimization_results['batch_size']}"),
                mo.ui.table(table_data, label="Performance Comparison"),
                mo.md("### 📊 Latency Distribution"),
                mo.ui.plotly(fig),
                mo.md("### ⚡ Speedup Analysis"),
                mo.ui.plotly(fig_speedup),
                mo.callout(
                    mo.md(f"""
                    **TensorRT delivers {speedup:.2f}x faster inference!**
                    - Latency reduced from {pytorch_res['mean_latency']:.2f}ms to {trt_res['mean_latency']:.2f}ms
                    - Throughput increased to {trt_res['throughput']:.1f} images/second
                    - Precision: {optimization_results['precision']}
                    """),
                    kind="success"
                )
            ])
        elif trt_res and 'error' in trt_res:
            mo.vstack([
                mo.md(f"### ⚠️ TensorRT Conversion Failed"),
                mo.ui.table(table_data, label="PyTorch Performance"),
                mo.ui.plotly(fig),
                mo.callout(
                    mo.md(f"**TensorRT Error**: {trt_res['error']}"),
                    kind="warn"
                )
            ])
        else:
            mo.vstack([
                mo.md(f"### 📊 PyTorch Baseline"),
                mo.ui.table(table_data, label="Performance Results"),
                mo.ui.plotly(fig),
                mo.callout(
                    mo.md("**Install TensorRT** to see optimization speedups"),
                    kind="info"
                )
            ])
    return


@app.cell
def __(mo):
    """Documentation and resources"""
    mo.md(
        """
        ---
        
        ## 🎯 TensorRT Optimization Techniques
        
        **What TensorRT Does**:
        1. **Layer Fusion**: Combines operations (conv + bn + relu → single kernel)
        2. **Precision Calibration**: INT8 quantization with minimal accuracy loss
        3. **Kernel Auto-Tuning**: Selects optimal CUDA kernels for your GPU
        4. **Dynamic Tensor Memory**: Reduces memory footprint
        5. **Multi-Stream Execution**: Overlaps computation and data transfer
        
        **Precision Modes**:
        - **FP32**: Full precision (baseline)
        - **FP16**: 2x faster on Tensor Cores, minimal accuracy impact
        - **INT8**: 4x faster, requires calibration, 1-2% accuracy drop
        
        **Performance Tips**:
        1. **Use FP16 on Volta+ GPUs**: Tensor Cores provide massive speedups
        2. **Batch Inference**: Larger batches = better GPU utilization
        3. **Profile First**: Identify bottlenecks before optimizing
        4. **Dynamic Shapes**: Use optimization profiles for variable input sizes
        5. **Keep Models on GPU**: Avoid CPU-GPU transfers
        
        **When to Use TensorRT**:
        - Production inference deployments
        - Real-time applications (video, robotics)
        - High-throughput serving
        - Edge deployment (Jetson devices)
        - Cost optimization (reduce GPU instances)
        
        ### 🚀 Production Deployment
        
        **Export TensorRT Engine**:
        ```python
        import torch_tensorrt
        
        # Compile model
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
            enabled_precisions={torch.float16},
        )
        
        # Save engine
        torch.jit.save(trt_model, "model_trt.ts")
        ```
        
        **Load and Use**:
        ```python
        # Load compiled model
        trt_model = torch.jit.load("model_trt.ts")
        
        # Run inference
        output = trt_model(input_tensor)
        ```
        
        **Run this notebook**:
        ```bash
        python tensorrt_optimization.py
        ```
        
        **Deploy as app**:
        ```bash
        marimo run tensorrt_optimization.py
        ```
        
        ### 📖 Resources
        - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
        - [Torch-TensorRT](https://pytorch.org/TensorRT/)
        - [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
        - [NGC Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt)
        - [Brev.dev Platform](https://brev.dev)
        """
    )
    return


if __name__ == "__main__":
    app.run()

