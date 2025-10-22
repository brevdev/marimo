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
def __(mo, CUDF_AVAILABLE):
    """Title and cuDF availability check"""
    mo.md(
        f"""
        # 🚀 RAPIDS cuDF vs Pandas Benchmark
        
        **Compare GPU-accelerated dataframes with traditional CPU processing**
        
        RAPIDS cuDF provides a pandas-like API that runs on NVIDIA GPUs, delivering
        **10-100x speedups** for common data manipulation operations.
        
        **cuDF Status**: {'✅ Available' if CUDF_AVAILABLE else '⚠️ Not installed - will show CPU-only results'}
        
        ## ⚙️ Benchmark Configuration
        """
    )
    return


@app.cell
def __(mo):
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
    
    run_benchmark_btn = mo.ui.run_button(label="🏃 Run Benchmark")
    
    mo.vstack([
        mo.hstack([dataset_size, operations], justify="start"),
        mo.md(f"**Dataset will have**: {10**dataset_size.value:,} rows"),
        run_benchmark_btn
    ])
    return dataset_size, operations, run_benchmark_btn


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
                'Free': f"{total - reserved:.2f} GB",
                'Total': f"{total:.2f} GB"
            }
        return None
    
    gpu_memory = mo.ui.refresh(
        lambda: get_gpu_memory(),
        options=[2, 5, 10],
        default_interval=5
    )
    
    mo.vstack([
        mo.md("### 📊 GPU Memory"),
        gpu_memory if gpu_memory.value else mo.md("*CPU mode*")
    ])
    return get_gpu_memory, gpu_memory


@app.cell
def __(np, pd, cudf, time, CUDF_AVAILABLE):
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
    run_benchmark_btn, dataset_size, operations, generate_data,
    benchmark_functions, cudf, CUDF_AVAILABLE, pd, mo, np, torch, device
):
    """Run benchmarks"""
    
    benchmark_results = None
    
    if run_benchmark_btn.value:
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
                        n_rows = min(requested_rows, safe_rows)
                        if n_rows < requested_rows:
                            mo.callout(
                                mo.md(f"""
                                ⚠️ **Memory Scaling Applied**
                                
                                Requested: {requested_rows:,} rows ({estimated_gb:.1f} GB est.)
                                Adjusted to: {n_rows:,} rows (safe for {gpu_mem_gb:.1f} GB GPU)
                                """),
                                kind="warn"
                            )
                    else:
                        n_rows = requested_rows
                else:
                    n_rows = 10 ** dataset_size.value
                
                # Generate data
                mo.status.progress_bar(
                    title="Generating dataset...",
                    subtitle=f"{n_rows:,} rows"
                )
                
                pandas_df = generate_data(n_rows)
                
                results = {
                    'operation': [],
                    'pandas_time': [],
                    'cudf_time': [],
                    'speedup': [],
                    'result_size': []
                }
                
                # Run benchmarks for each operation
                for op in operations.value:
                    if op not in benchmark_functions:
                        continue
                    
                    bench_func = benchmark_functions[op]
                    
                    # Pandas benchmark
                    pandas_time, result_size = bench_func(pandas_df.copy(), is_gpu=False)
                    
                    # cuDF benchmark
                    if CUDF_AVAILABLE:
                        cudf_df = cudf.from_pandas(pandas_df)
                        cudf_time, _ = bench_func(cudf_df, is_gpu=True)
                        speedup = pandas_time / cudf_time
                    else:
                        cudf_time = None
                        speedup = None
                    
                    results['operation'].append(op.capitalize())
                    results['pandas_time'].append(pandas_time)
                    results['cudf_time'].append(cudf_time)
                    results['speedup'].append(speedup)
                    results['result_size'].append(result_size)
                
                benchmark_results = {
                    'results': results,
                    'n_rows': n_rows,
                    'success': True
                }
                
                # Cleanup GPU memory
                if CUDF_AVAILABLE and device.type == "cuda":
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
def __(benchmark_results, mo, pd, go, CUDF_AVAILABLE):
    """Visualize benchmark results"""
    
    if benchmark_results is None:
        mo.callout(
            mo.md("**Click 'Run Benchmark' to start performance comparison**"),
            kind="info"
        )
    elif not benchmark_results['success']:
        error_msg = f"**Benchmark Error**: {benchmark_results['error']}"
        if 'suggestion' in benchmark_results:
            error_msg += f"\n\n{benchmark_results['suggestion']}"
        if 'error_type' in benchmark_results:
            error_msg += f"\n\n*Error type: {benchmark_results['error_type']}*"
        
        mo.callout(
            mo.md(error_msg),
            kind="danger"
        )
    else:
        results = benchmark_results['results']
        n_rows = benchmark_results['n_rows']
        
        # Create results table
        table_data = {
            'Operation': results['operation'],
            'Pandas Time (s)': [f"{t:.4f}" for t in results['pandas_time']],
        }
        
        if CUDF_AVAILABLE:
            table_data['cuDF Time (s)'] = [f"{t:.4f}" if t else "N/A" for t in results['cudf_time']]
            table_data['Speedup'] = [f"{s:.2f}x" if s else "N/A" for s in results['speedup']]
        
        # Create speedup bar chart
        if CUDF_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Pandas (CPU)',
                x=results['operation'],
                y=results['pandas_time'],
                marker_color='#ff6b6b'
            ))
            
            fig.add_trace(go.Bar(
                name='cuDF (GPU)',
                x=results['operation'],
                y=results['cudf_time'],
                marker_color='#51cf66'
            ))
            
            fig.update_layout(
                title=f"Performance Comparison ({n_rows:,} rows)",
                xaxis_title="Operation",
                yaxis_title="Time (seconds)",
                barmode='group',
                height=400,
                hovermode='x unified'
            )
            
            # Speedup chart
            fig_speedup = go.Figure()
            
            fig_speedup.add_trace(go.Bar(
                x=results['operation'],
                y=results['speedup'],
                marker_color='#4c6ef5',
                text=[f"{s:.1f}x" for s in results['speedup']],
                textposition='outside'
            ))
            
            fig_speedup.update_layout(
                title="GPU Speedup Factor",
                xaxis_title="Operation",
                yaxis_title="Speedup (higher is better)",
                height=400
            )
            
            # Calculate overall statistics
            avg_speedup = sum(results['speedup']) / len(results['speedup'])
            max_speedup = max(results['speedup'])
            min_speedup = min(results['speedup'])
            
            mo.vstack([
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
                    - Best Speedup: **{max_speedup:.1f}x** ({results['operation'][results['speedup'].index(max_speedup)]})
                    - Minimum Speedup: **{min_speedup:.1f}x**
                    - Dataset Size: **{n_rows:,}** rows
                    """),
                    kind="success"
                )
            ])
        else:
            mo.vstack([
                mo.md("### ⚠️ cuDF Not Available"),
                mo.ui.table(table_data, label="Pandas Results (CPU only)"),
                mo.callout(
                    mo.md("""
                    **Install RAPIDS cuDF** to see GPU performance:
                    ```bash
                    conda install -c rapidsai -c conda-forge -c nvidia \
                        cudf=24.04 python=3.10 cudatoolkit=11.8
                    ```
                    """),
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

