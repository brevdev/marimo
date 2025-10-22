"""rapids_cudf_benchmark.py

RAPIDS cuDF vs Pandas Performance Comparison
=============================================

Interactive benchmark comparing CPU-based Pandas with GPU-accelerated cuDF for
dataframe operations. Demonstrates the massive speedups possible with RAPIDS on
NVIDIA GPUs for data manipulation, aggregation, and analysis.

Features:
- Interactive dataset size selection (1K to 100M rows)
- Real-time performance comparison across multiple operations
- Speedup visualization with interactive charts
- Memory usage tracking for both CPU and GPU
- Multiple operation types: filtering, groupby, joins, sorting

Requirements:
- NVIDIA GPU with 4GB+ VRAM (works on all data center GPUs)
- Tested on: L40S (48GB), A100 (40/80GB), H100 (80GB), H200 (141GB), B200 (180GB), RTX PRO 6000 (48GB)
- CUDA 11.4+
- RAPIDS cuDF installed (optional, falls back to pandas)
- Dataset size automatically scales with available memory
- Single GPU only (cuDF uses GPU 0)

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
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from typing import Dict, List, Tuple, Optional
    import time
    import subprocess
    import torch
    
    # Try importing cuDF
    try:
        import cudf
        CUDF_AVAILABLE = True
    except ImportError:
        CUDF_AVAILABLE = False
        cudf = None
    
    return (
        mo, np, pd, go, px, Dict, List, Tuple, Optional,
        time, subprocess, torch, cudf, CUDF_AVAILABLE
    )


@app.cell
def __(mo, CUDF_AVAILABLE, subprocess):
    """Auto-install cuDF if not available"""
    
    # Initialize at cell level - keep same name for consistency
    cudf_available = CUDF_AVAILABLE
    install_result = mo.md("")
    
    if not CUDF_AVAILABLE:
        with mo.status.spinner(title="Installing cuDF for GPU acceleration...", subtitle="This takes 2-3 minutes"):
            try:
                result = subprocess.run(
                    ["pip", "install", "cudf-cu12", "--extra-index-url=https://pypi.nvidia.com"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    install_result = mo.callout(
                        mo.md("✅ **cuDF installed!** Restart the notebook to activate GPU acceleration."),
                        kind="success"
                    )
                    # Try importing again (may need kernel restart)
                    try:
                        import cudf as _cudf
                        cudf_available = True
                    except ImportError:
                        cudf_available = False
                else:
                    install_result = mo.callout(
                        mo.md(f"⚠️ **Could not auto-install cuDF**. Install manually:\n```bash\npip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com\n```"),
                        kind="warn"
                    )
            except Exception as e:
                install_result = mo.callout(
                    mo.md(f"⚠️ **Install skipped**: {str(e)[:100]}. Running CPU-only mode."),
                    kind="warn"
                )
    
    return cudf_available, install_result


@app.cell
def __(mo, cudf_available, install_result):
    """Title and cuDF availability check"""
    mo.vstack([
        mo.md(
            f"""
            # 🚀 RAPIDS cuDF vs Pandas Benchmark
            
            **Compare GPU-accelerated dataframes with traditional CPU processing**
            
            RAPIDS cuDF provides a pandas-like API that runs on NVIDIA GPUs, delivering
            **10-100x speedups** for common data manipulation operations.
            
            **cuDF Status**: {'✅ Available' if cudf_available else '⚠️ Not installed - will show CPU-only results'}
            
            ## ⚙️ Benchmark Configuration
            """
        ),
        install_result
    ])
    return


@app.cell
def __(mo, cudf_available):
    """Interactive benchmark controls"""
    dataset_size = mo.ui.slider(
        start=3, stop=7, step=1, value=5,
        label="Dataset Size (10^x rows)", show_value=True
    )
    
    operations = mo.ui.multiselect(
        options=['filter', 'groupby', 'join', 'sort', 'rolling', 'merge'],
        value=['filter', 'groupby', 'join', 'sort'],
        label="Operations to Benchmark"
    )
    
    # CPU/GPU mode toggle (only if cuDF available)
    if cudf_available:
        mode_toggle = mo.ui.dropdown(
            options={'both': 'CPU vs GPU', 'cpu': 'CPU Only', 'gpu': 'GPU Only'},
            value='both',
            label="Benchmark Mode"
        )
    else:
        mode_toggle = mo.ui.dropdown(
            options={'cpu': 'CPU Only (cuDF not installed)'},
            value='cpu',
            label="Benchmark Mode"
        )
    
    run_benchmark_btn = mo.ui.run_button(label="🏃 Run Benchmark")
    
    return dataset_size, operations, mode_toggle, run_benchmark_btn


@app.cell
def __(mo, dataset_size, operations, mode_toggle, run_benchmark_btn, cudf_available):
    """Display benchmark controls"""
    mo.vstack([
        mo.hstack([dataset_size, operations], justify="start"),
        mo.hstack([mode_toggle, run_benchmark_btn], justify="start"),
        mo.md(f"**Dataset will have**: {10**dataset_size.value:,} rows"),
        mo.callout(
            mo.md(f"""
            **Mode**: {mode_toggle.value}
            
            {'✅ cuDF available - you can compare CPU vs GPU!' if cudf_available else '⚠️ cuDF not installed - running CPU-only benchmarks'}
            
            {'💡 Tip: Try "CPU vs GPU" mode to see the speedup!' if cudf_available else '💡 Restart the notebook after cuDF installs!'}
            """),
            kind="info" if cudf_available else "warn"
        )
    ])
    return


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
    
    # Stop execution if no GPU available (RAPIDS requires GPU)
    if not gpu_info['available']:
        mo.stop(
            True,
            mo.callout(
                mo.md(f"""
                ⚠️ **No GPU Detected**
                
                This notebook requires an NVIDIA GPU for RAPIDS cuDF benchmarking.
                
                **Error**: {gpu_info.get('error', 'Unknown')}
                
                **Troubleshooting**:
                - Run `nvidia-smi` to verify GPU is detected
                - Check CUDA driver installation
                - Ensure PyTorch has CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
                
                **Note**: cuDF requires GPU to demonstrate speedup vs Pandas.
                """),
                kind="danger"
            )
        )
    
    # Display GPU info
    mo.callout(
        mo.vstack([
            mo.md("**✅ GPU Detected for RAPIDS**"),
            mo.ui.table(gpu_info['gpus'])
        ]),
        kind="success"
    )
    
    device = torch.device("cuda")
    
    return get_gpu_info, gpu_info, device


@app.cell
def __(mo):
    """GPU Memory Monitor - Refresh trigger"""
    gpu_memory_refresh = mo.ui.refresh(default_interval="5s")
    return gpu_memory_refresh,


@app.cell
def __(mo, device, torch, gpu_memory_refresh, Optional, Dict):
    """GPU Memory Display"""
    # Trigger refresh
    _ = gpu_memory_refresh.value
    
    def get_gpu_memory() -> Optional[Dict]:
        """Get current GPU memory usage"""
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'Allocated': f"{allocated:.2f} GB",
                'Reserved': f"{reserved:.2f} GB",
                'Free': f"{total - reserved:.2f} GB",
                'Total': f"{total:.2f} GB"
            }
        return None
    
    gpu_memory_data = get_gpu_memory()
    
    mo.vstack([
        mo.md("### 📊 GPU Memory"),
        mo.ui.table(gpu_memory_data) if gpu_memory_data else mo.md("*CPU mode*"),
        gpu_memory_refresh
    ])
    return get_gpu_memory, gpu_memory_data


@app.cell
def __(np, pd, cudf, time, cudf_available):
    """Benchmark functions"""
    
    def generate_data(n_rows: int) -> pd.DataFrame:
        """Generate synthetic dataset"""
        np.random.seed(42)
        data = {
            'id': np.arange(n_rows),
            'value': np.random.randn(n_rows),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'group': np.random.randint(0, 100, n_rows),
            'timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='1s')
        }
        return pd.DataFrame(data)
    
    def benchmark_filter(df, is_gpu: bool = False) -> Tuple[float, int]:
        """Benchmark filtering operation"""
        start = time.time()
        result = df[df['value'] > 0]
        if is_gpu:
            result = result.to_pandas()  # Force computation
        elapsed = time.time() - start
        return elapsed, len(result)
    
    def benchmark_groupby(df, is_gpu: bool = False) -> Tuple[float, int]:
        """Benchmark groupby aggregation"""
        start = time.time()
        result = df.groupby('category')['value'].mean()
        if is_gpu:
            result = result.to_pandas()  # Force computation
        elapsed = time.time() - start
        return elapsed, len(result)
    
    def benchmark_join(df, is_gpu: bool = False) -> Tuple[float, int]:
        """Benchmark join operation"""
        # Create second dataframe
        if is_gpu:
            df2 = cudf.DataFrame({
                'group': range(100),
                'multiplier': np.random.randn(100)
            })
        else:
            df2 = pd.DataFrame({
                'group': range(100),
                'multiplier': np.random.randn(100)
            })
        
        start = time.time()
        result = df.merge(df2, on='group', how='left')
        if is_gpu:
            result = result.to_pandas()  # Force computation
        elapsed = time.time() - start
        return elapsed, len(result)
    
    def benchmark_sort(df, is_gpu: bool = False) -> Tuple[float, int]:
        """Benchmark sorting operation"""
        start = time.time()
        result = df.sort_values('value')
        if is_gpu:
            result = result.to_pandas()  # Force computation
        elapsed = time.time() - start
        return elapsed, len(result)
    
    def benchmark_rolling(df, is_gpu: bool = False) -> Tuple[float, int]:
        """Benchmark rolling window operation"""
        start = time.time()
        result = df.sort_values('timestamp')['value'].rolling(window=100).mean()
        if is_gpu:
            result = result.to_pandas()  # Force computation
        elapsed = time.time() - start
        return elapsed, len(result)
    
    def benchmark_merge(df, is_gpu: bool = False) -> Tuple[float, int]:
        """Benchmark complex merge with aggregation"""
        start = time.time()
        agg = df.groupby('group').agg({'value': ['mean', 'std', 'min', 'max']})
        if is_gpu:
            agg = agg.to_pandas()  # Force computation
        elapsed = time.time() - start
        return elapsed, len(agg)
    
    benchmark_functions = {
        'filter': benchmark_filter,
        'groupby': benchmark_groupby,
        'join': benchmark_join,
        'sort': benchmark_sort,
        'rolling': benchmark_rolling,
        'merge': benchmark_merge
    }
    
    return (
        generate_data, benchmark_filter, benchmark_groupby, 
        benchmark_join, benchmark_sort, benchmark_rolling,
        benchmark_merge, benchmark_functions
    )


@app.cell
def __(
    run_benchmark_btn, dataset_size, operations, mode_toggle, generate_data,
    benchmark_functions, cudf, cudf_available, pd, mo, np, torch, device
):
    """Run benchmarks"""
    
    benchmark_results = None
    
    # Check if button was clicked (value increments on each click)
    if run_benchmark_btn.value > 0:
        print(f"✅ DEBUG: Button clicked! Running benchmark with {len(operations.value)} operations")
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        with mo.status.spinner(
            title="🔄 Running RAPIDS cuDF Benchmarks...",
            subtitle=f"Operations: {', '.join(operations.value)}, Dataset: {10**dataset_size.value:,} rows"
        ):
            try:
                # Check GPU memory and adjust dataset size if needed
                if device.type == "cuda":
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    requested_rows = 10 ** dataset_size.value
                    
                    # Estimate memory: ~50 bytes per row (conservative)
                    estimated_gb = (requested_rows * 50) / 1024**3
                    
                    if estimated_gb > gpu_mem_gb * 0.7:  # Don't use more than 70% of GPU memory
                        # Scale down to safe size
                        safe_rows = int((gpu_mem_gb * 0.7 * 1024**3) / 50)
                        _n_rows = min(requested_rows, safe_rows)
                        if _n_rows < requested_rows:
                            mo.callout(
                                mo.md(f"""
                                ⚠️ **Memory Scaling Applied**
                                
                                Requested: {requested_rows:,} rows ({estimated_gb:.1f} GB est.)
                                Adjusted to: {_n_rows:,} rows (safe for {gpu_mem_gb:.1f} GB GPU)
                                """),
                                kind="warn"
                            )
                    else:
                        _n_rows = requested_rows
                else:
                    _n_rows = 10 ** dataset_size.value
                
                # Generate data
                mo.status.progress_bar(
                    title="Generating dataset...",
                    subtitle=f"{_n_rows:,} rows"
                )
                
                pandas_df = generate_data(_n_rows)
                
                _results = {
                    'operation': [],
                    'pandas_time': [],
                    'cudf_time': [],
                    'speedup': [],
                    'result_size': []
                }
                
                # Run benchmarks for each operation based on mode
                _mode = mode_toggle.value
                
                for op in operations.value:
                    if op not in benchmark_functions:
                        continue
                    
                    bench_func = benchmark_functions[op]
                    
                    # Pandas benchmark (run if mode is 'cpu' or 'both')
                    if _mode in ['cpu', 'both']:
                        pandas_time, result_size = bench_func(pandas_df.copy(), is_gpu=False)
                    else:
                        pandas_time, result_size = None, 0
                    
                    # cuDF benchmark (run if mode is 'gpu' or 'both' AND cuDF available)
                    if _mode in ['gpu', 'both'] and cudf_available:
                        cudf_df = cudf.from_pandas(pandas_df)
                        cudf_time, _ = bench_func(cudf_df, is_gpu=True)
                        speedup = pandas_time / cudf_time if pandas_time else None
                    else:
                        cudf_time = None
                        speedup = None
                    
                    _results['operation'].append(op.capitalize())
                    _results['pandas_time'].append(pandas_time)
                    _results['cudf_time'].append(cudf_time)
                    _results['speedup'].append(speedup)
                    _results['result_size'].append(result_size)
                
                benchmark_results = {
                    'results': _results,
                    'n_rows': _n_rows,
                    'mode': _mode,
                    'success': True
                }
                
                # Cleanup GPU memory
                if cudf_available and device.type == "cuda":
                    del cudf_df
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                # Handle GPU OOM explicitly
                torch.cuda.empty_cache()
                benchmark_results = {
                    'error': 'GPU Out of Memory',
                    'suggestion': f"""
**GPU ran out of memory!**

**Current settings**:
- Dataset size: {10**dataset_size.value:,} rows
- Operations: {', '.join(operations.value)}

**Try these solutions**:
1. Reduce dataset size (try {dataset_size.value - 1} or {dataset_size.value - 2})
2. Run fewer operations at once
3. Close other GPU applications

**GPU**: {torch.cuda.get_device_properties(0).name} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)
                    """,
                    'success': False
                }
            except Exception as e:
                benchmark_results = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'success': False
                }
    
    return benchmark_results,


@app.cell
def __(benchmark_results, mo, pd, go, cudf_available, run_benchmark_btn):
    """Visualize benchmark results"""
    
    # Show status based on button clicks and results
    if run_benchmark_btn.value > 0:
        if benchmark_results and benchmark_results.get('success'):
            status = mo.callout(
                mo.md(f"✅ **Benchmark completed!** Mode: `{benchmark_results.get('mode', 'unknown')}`"),
                kind="success"
            )
        elif benchmark_results and not benchmark_results.get('success'):
            status = mo.callout(
                mo.md(f"❌ **Benchmark failed:** {benchmark_results.get('error', 'unknown')}"),
                kind="danger"
            )
        else:
            status = mo.callout(
                mo.md("⏳ **Running benchmark...** This may take a moment."),
                kind="info"
            )
    else:
        status = None
    
    if benchmark_results is None:
        # Show initial message before any benchmark runs
        return mo.callout(
            mo.md("**Click 'Run Benchmark' to start performance comparison**"),
            kind="info"
        )
    elif not benchmark_results['success']:
        # Show error message
        error_msg = f"**Benchmark Error**: {benchmark_results['error']}"
        if 'suggestion' in benchmark_results:
            error_msg += f"\n\n{benchmark_results['suggestion']}"
        if 'error_type' in benchmark_results:
            error_msg += f"\n\n*Error type: {benchmark_results['error_type']}*"
        
        return mo.callout(
            mo.md(error_msg),
            kind="danger"
        )
    else:
        # Show benchmark results
        _results = benchmark_results['results']
        _n_rows = benchmark_results['n_rows']
        
        # Create results table
        _mode = benchmark_results.get('mode', 'cpu')
        table_data = {'Operation': _results['operation']}
        
        # Add Pandas times if they were run
        if _mode in ['cpu', 'both'] and any(_results['pandas_time']):
            table_data['Pandas Time (s)'] = [f"{t:.4f}" if t else "N/A" for t in _results['pandas_time']]
        
        # Add cuDF times if they were run
        if _mode in ['gpu', 'both'] and cudf_available and any(_results['cudf_time']):
            table_data['cuDF Time (s)'] = [f"{t:.4f}" if t else "N/A" for t in _results['cudf_time']]
        
        # Add speedup if both were run
        if _mode == 'both' and any(_results['speedup']):
            table_data['Speedup'] = [f"{s:.2f}x" if s else "N/A" for s in _results['speedup']]
        
        # Create performance visualization
        fig = go.Figure()
        
        # Add Pandas performance if run
        if _mode in ['cpu', 'both'] and any(_results['pandas_time']):
            pandas_times = [t if t else 0 for t in _results['pandas_time']]
            fig.add_trace(go.Bar(
                name='Pandas (CPU)',
                x=_results['operation'],
                y=pandas_times,
                marker_color='#ff6b6b',
                text=[f"{t:.3f}s" if t else "N/A" for t in _results['pandas_time']],
                textposition='outside'
            ))
        
        # Add cuDF if run
        if _mode in ['gpu', 'both'] and cudf_available and any(_results['cudf_time']):
            cudf_times = [t if t else 0 for t in _results['cudf_time']]
            fig.add_trace(go.Bar(
                name='cuDF (GPU)',
                x=_results['operation'],
                y=cudf_times,
                marker_color='#51cf66',
                text=[f"{t:.3f}s" if t else "N/A" for t in _results['cudf_time']],
                textposition='outside'
            ))
        
        _title_suffix = {
            'cpu': ' - CPU Only',
            'gpu': ' - GPU Only',
            'both': ' - CPU vs GPU'
        }.get(_mode, '')
        
        fig.update_layout(
            title=f"Performance Comparison ({_n_rows:,} rows){_title_suffix}",
            xaxis_title="Operation",
            yaxis_title="Time (seconds)",
            barmode='group',
            height=400,
            hovermode='x unified'
        )
        
        # Show results
        if cudf_available and _mode in ['both', 'gpu']:
            # Speedup chart (only with cuDF)
            fig_speedup = go.Figure()
            
            fig_speedup.add_trace(go.Bar(
                x=_results['operation'],
                y=_results['speedup'],
                marker_color='#4c6ef5',
                text=[f"{s:.1f}x" for s in _results['speedup']],
                textposition='outside'
            ))
            
            fig_speedup.update_layout(
                title="GPU Speedup Factor",
                xaxis_title="Operation",
                yaxis_title="Speedup (higher is better)",
                height=400
            )
            
            # Calculate overall statistics
            avg_speedup = sum(_results['speedup']) / len(_results['speedup'])
            max_speedup = max(_results['speedup'])
            min_speedup = min(_results['speedup'])
            
            return mo.vstack([
                status if status else mo.md(""),
                mo.md("### ✅ Benchmark Complete!"),
                mo.ui.table(table_data, label="Performance Results"),
                mo.md("### 📊 Execution Time Comparison"),
                mo.ui.plotly(fig),
                mo.md("### ⚡ Speedup Analysis"),
                mo.ui.plotly(fig_speedup),
                mo.callout(
                    mo.md(f"""
                    **Performance Summary**:
                    - Average Speedup: **{avg_speedup:.1f}x**
                    - Best Speedup: **{max_speedup:.1f}x** ({_results['operation'][_results['speedup'].index(max_speedup)]})
                    - Minimum Speedup: **{min_speedup:.1f}x**
                    - Dataset Size: **{_n_rows:,}** rows
                    """),
                    kind="success"
                )
            ])
        else:
            # CPU-only mode
            return mo.vstack([
                status if status else mo.md(""),
                mo.md("### ✅ Benchmark Complete (CPU-Only Mode)!"),
                mo.callout(
                    mo.md(f"""
                    ⚠️ **Running in CPU-only mode** (cuDF not installed)
                    
                    Install cuDF to see GPU-accelerated performance:
                    ```bash
                    pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
                    ```
                    """),
                    kind="warn"
                ),
                mo.ui.table(table_data, label="Performance Results"),
                mo.md("### 📊 Pandas Performance (CPU)"),
                mo.ui.plotly(fig),
                mo.callout(
                    mo.md(f"""
                    **Pandas Performance Summary**:
                    - Dataset Size: **{_n_rows:,}** rows
                    - Operations Tested: **{len(_results['operation'])}**
                    - Total Time: **{sum(_results['pandas_time']):.2f}s**
                    - Average Time per Operation: **{sum(_results['pandas_time'])/len(_results['pandas_time']):.3f}s**
                    
                    💡 With cuDF (GPU), these operations could be **10-100x faster**!
                    """),
                    kind="info"
                )
            ])


@app.cell
def __(mo):
    """Documentation and resources"""
    mo.md(
        """
        ---
        
        ## 🎯 Key Takeaways
        
        **When to Use cuDF**:
        - Large datasets (1M+ rows)
        - Repetitive data transformations
        - ETL pipelines
        - Real-time analytics
        - Exploratory data analysis at scale
        
        **Performance Tips**:
        1. **Batch Operations**: Combine multiple operations to minimize CPU-GPU transfers
        2. **Keep Data on GPU**: Avoid frequent `.to_pandas()` calls
        3. **Use Native cuDF Functions**: Some operations are optimized differently than pandas
        4. **Memory Management**: Monitor GPU memory, use chunking for very large datasets
        
        **RAPIDS Ecosystem**:
        - **cuDF**: GPU DataFrames (pandas-like)
        - **cuML**: GPU Machine Learning (sklearn-like)
        - **cuGraph**: GPU Graph Analytics (NetworkX-like)
        - **cuSpatial**: GPU Spatial Analytics
        - **cuSignal**: GPU Signal Processing
        
        ### 🚀 Production Deployment
        
        **Run as Script**:
        ```bash
        python rapids_cudf_benchmark.py
        ```
        
        **Deploy as App**:
        ```bash
        marimo run rapids_cudf_benchmark.py
        ```
        
        **Integrate in Pipeline**:
        ```python
        import cudf
        
        # Read from various sources
        df = cudf.read_csv('large_dataset.csv')
        df = cudf.read_parquet('data.parquet')
        
        # Process at GPU speed
        result = df.groupby('category').agg({'value': 'mean'})
        
        # Write results
        result.to_parquet('output.parquet')
        ```
        
        ### 📖 Resources
        - [RAPIDS Documentation](https://docs.rapids.ai/)
        - [cuDF API Reference](https://docs.rapids.ai/api/cudf/stable/)
        - [10 Minutes to cuDF](https://docs.rapids.ai/api/cudf/stable/user_guide/10min.html)
        - [RAPIDS Notebooks](https://github.com/rapidsai/notebooks)
        - [Brev.dev Platform](https://brev.dev)
        """
    )
    return


if __name__ == "__main__":
    app.run()

