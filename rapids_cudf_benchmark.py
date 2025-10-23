"""rapids_cudf_benchmark.py

RAPIDS cuDF vs Pandas Performance Comparison
=============================================

Interactive benchmark comparing CPU-based Pandas with GPU-accelerated cuDF for
dataframe operations. Demonstrates the massive speedups possible with RAPIDS on
NVIDIA GPUs for data manipulation, aggregation, and analysis.

Features:
- Interactive dataset size selection (1K to 100M rows)
- Performance comparison across multiple operations (filter, groupby, join, sort)
- Speedup visualization with interactive Plotly charts
- GPU performance timeline charts (utilization, memory, temperature over time)
- Multi-GPU monitoring and support
- Console progress output during benchmark execution

Requirements:
- NVIDIA GPU with 4GB+ VRAM (works on all NVIDIA datacenter GPUs)
- Tested on: L4, L40, L40S, A10, A100, H100, H200, B100, B200, GH200, GB200
- Also works on: RTX 4000 Ada, RTX 5000 Ada, RTX 6000 Ada, and other NVIDIA professional GPUs
- CUDA 11.4+
- RAPIDS cuDF installed (optional, falls back to pandas)
- Dataset size automatically scales with available memory
- Multi-GPU aware: Monitors all GPUs, uses GPU 0 for compute by default
- For multi-GPU data distribution: Install dask-cudf and dask-cuda

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
    import threading
    from queue import Queue
    
    # Try importing cuDF
    try:
        import cudf
        CUDF_AVAILABLE = True
    except ImportError:
        CUDF_AVAILABLE = False
        cudf = None
    
    # Try importing Dask-cuDF for multi-GPU support
    try:
        import dask_cudf
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        DASK_CUDF_AVAILABLE = True
    except ImportError:
        DASK_CUDF_AVAILABLE = False
        dask_cudf = None
        LocalCUDACluster = None
        Client = None
    
    return (
        mo, np, pd, go, px, Dict, List, Tuple, Optional,
        time, subprocess, torch, cudf, CUDF_AVAILABLE,
        dask_cudf, LocalCUDACluster, Client, DASK_CUDF_AVAILABLE,
        threading, Queue
    )


@app.cell
def __(mo, CUDF_AVAILABLE, subprocess):
    """Auto-install cuDF if not available and ensure it's properly imported"""
    
    # Always try to import cuDF fresh, don't rely on cell-1's import
    cudf_module = None
    cudf_available = False
    install_result = mo.md("")
    
    # Try importing cuDF first
    try:
        import cudf as cudf_module
        cudf_available = True
        print("✅ cuDF already available and imported")
    except ImportError:
        cudf_module = None
        cudf_available = False
        print("⚠️ cuDF not found, attempting installation...")
    
    if not cudf_available:
        with mo.status.spinner(title="Installing cuDF for GPU acceleration...", subtitle="This takes 2-3 minutes"):
            try:
                result = subprocess.run(
                    ["pip", "install", "cudf-cu12", "--extra-index-url=https://pypi.nvidia.com"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # Try importing the newly installed cuDF
                    try:
                        import cudf as cudf_module
                        cudf_available = True
                        install_result = mo.callout(
                            mo.md("✅ **cuDF installed and imported successfully!** GPU acceleration is now active."),
                            kind="success"
                        )
                    except ImportError:
                        cudf_available = False
                        cudf_module = None
                        install_result = mo.callout(
                            mo.md("✅ **cuDF installed!** Please restart the notebook to activate GPU acceleration."),
                            kind="success"
                        )
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
    else:
        # cuDF was already available, keep the original module
        cudf_module = cudf if CUDF_AVAILABLE else None
    
    return cudf_available, cudf_module, install_result


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
            """
        ),
        install_result
    ])
    return


@app.cell
def __(mo, cudf_available):
    """Interactive benchmark controls"""
    dataset_size = mo.ui.slider(
        start=3, stop=8, step=1, value=6,  # Default to 1M rows, max 100M to push GPU to 100%
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
    
    run_benchmark_btn = mo.ui.run_button(label="🏃 Run Benchmark", kind="success")
    
    return dataset_size, operations, mode_toggle, run_benchmark_btn


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
    
    # Count GPUs
    num_gpus = len(gpu_info['gpus'])
    multi_gpu_msg = ""
    if num_gpus > 1:
        multi_gpu_msg = f"\n\n💡 **{num_gpus} GPUs detected!** This notebook monitors all GPUs. cuDF will use GPU 0 by default. For true multi-GPU data distribution, use Dask-cuDF (install: `pip install dask-cudf dask-cuda`)."
    
    # Display GPU info
    mo.callout(
        mo.vstack([
            mo.md(f"**✅ {num_gpus} GPU{'s' if num_gpus > 1 else ''} Detected for RAPIDS**{multi_gpu_msg}"),
            mo.ui.table(gpu_info['gpus'])
        ]),
        kind="success"
    )
    
    device = torch.device("cuda")
    
    return get_gpu_info, gpu_info, device, num_gpus


@app.cell
def __(mo):
    """Educational: Why GPU-Accelerated DataFrames?"""
    mo.vstack([
        mo.md("---"),
        mo.md("## 💡 Why GPU-Accelerated DataFrames?"),
        mo.accordion({
            "Click to expand educational content": mo.md("""
### The Problem with Traditional Data Processing
- CPUs process data **sequentially** (one row at a time, even with multiple cores)
- Moving data between CPU cores is slow due to memory bandwidth limits
- Python loops are especially slow due to interpreter overhead

### How GPUs Solve This
- GPUs have **thousands of cores** (vs CPU's ~10-100 cores)
- **Massively parallel**: Process thousands of rows simultaneously
- **High memory bandwidth**: 10-20x faster than CPU memory (900 GB/s vs 50 GB/s)
- **SIMD operations**: Same instruction on many data points at once

### When GPU Acceleration Helps Most
- ✅ **Large datasets** (1M+ rows) - enough work to saturate GPU
- ✅ **Vectorized operations** (filter, groupby, join, math) - GPU-optimized kernels
- ✅ **Batch processing** - amortize data transfer costs
- ❌ **Row-by-row `.apply()` with custom functions** - falls back to slow CPU execution
- ❌ **Small datasets** (<10K rows) - CPU faster due to transfer overhead

> **Note**: Python works great with GPUs! cuDF, PyTorch, and RAPIDS are all Python. 
> The limitation is row-by-row operations, not Python itself. Stick to vectorized operations (built-in functions) for speed.

### The 1M Row Threshold

**Below ~1M rows**, CPU is often faster because:
1. **Data transfer overhead**: Moving data to GPU takes ~1ms
2. **Kernel launch overhead**: Starting GPU kernels takes time
3. **CPU caches work well**: Small data fits in L1/L2/L3 cache

**Above 1M rows**:
- Data no longer fits in CPU cache
- GPU's parallelism overcomes overhead
- 10-50x speedups become common
            """)
        }),
        mo.md("---")
    ])
    return


@app.cell
def __(mo, dataset_size, operations, mode_toggle, cudf_available):
    """Display benchmark controls (without run button - that comes after education)"""
    mo.vstack([
        mo.md("## ⚙️ Benchmark Configuration"),
        mo.hstack([dataset_size, operations], justify="start"),
        mo.hstack([mode_toggle], justify="start"),
        mo.md(f"**Dataset will have**: {10**dataset_size.value:,} rows"),
        mo.callout(
            mo.md(f"""
            **Mode**: {mode_toggle.value}
            
            {'✅ cuDF available - you can compare CPU vs GPU!' if cudf_available else '⚠️ cuDF not installed - running CPU-only benchmarks'}
            
            {'💡 **Pro tip**: Try 10M-100M rows (slider position 7-8) to push GPU to 100% utilization and see 20-50x speedup!' if cudf_available else '💡 Restart the notebook after cuDF installs!'}
            """),
            kind="info" if cudf_available else "warn"
        )
    ])
    return


@app.cell
def __(mo):
    """Educational: Understanding Benchmark Operations"""
    mo.vstack([
        mo.md("---"),
        mo.md("## 🔬 Understanding the Benchmark Operations"),
        mo.accordion({
            "Click to expand educational content": mo.md("""
### Filter Operation
**What it does**: Select rows meeting a condition (e.g., `value > 0.5`)

**Why it's fast on GPU:**
- **Parallel comparison**: GPU checks all rows simultaneously
- **Predicate evaluation**: Optimized CUDA kernels for comparisons
- **Memory coalescing**: Sequential memory access pattern
- **Result**: 10-20x speedup on large datasets

**CPU vs GPU difference:**
- CPU: Loop through rows, check condition (sequential)
- GPU: All cores check different rows at once (parallel)

### GroupBy-Aggregation
**What it does**: Group rows by category, compute statistics (e.g., mean per group)

**Why it's challenging:**
- **Irregular memory access**: Groups aren't contiguous in memory
- **Synchronization**: Need to combine results from different groups
- **Variable group sizes**: Some groups have 10 rows, others have 1M

**Why GPU excels here:**
- **Hash-based grouping**: Parallel hash computation
- **Atomic operations**: GPU hardware supports atomic adds (safe parallel updates)
- **Two-phase algorithm**: First count group sizes, then aggregate
- **Result**: 20-50x speedup on datasets with many groups

**This is where GPUs shine brightest!**

### Join Operation
**What it does**: Combine two dataframes based on common key

**Why it's memory-intensive:**
- **Build hash table**: First dataframe → hash table (O(n) space)
- **Probe hash table**: Second dataframe lookups (O(m) time)
- **Output materialization**: Create result dataframe

**GPU advantages:**
- **Parallel hash table build**: All rows hashed at once
- **High memory bandwidth**: 900 GB/s vs CPU's 50 GB/s
- **Parallel probing**: All lookups happen simultaneously
- **Result**: 15-40x speedup on large joins

**Why this matters for data science:**
- Joins are ubiquitous in data pipelines
- Often the bottleneck in ETL workflows
- GPU acceleration makes interactive analysis possible

### Sort Operation
**What it does**: Order rows by column value

**Why sorting is hard to parallelize:**
- **Dependencies**: Can't sort independently (results affect each other)
- **Comparison-based**: Need O(n log n) comparisons
- **Cache-unfriendly**: Random memory access patterns

**GPU sorting strategy:**
- **Radix sort**: Digit-by-digit sorting (GPU-friendly)
- **Merge sort**: Parallel merge of sorted chunks
- **Bitonic sort**: Network of compare-swap operations
- **Result**: 5-15x speedup (lower than other ops due to synchronization)

**Why speedup is lower:**
- Sorting has more dependencies than other operations
- More GPU synchronization points required
- Memory access patterns less optimal
            """)
        }),
        mo.md("---")
    ])
    return


@app.cell
def __(mo, run_benchmark_btn):
    """Display run benchmark button after educational content"""
    mo.vstack([
        mo.md("## 🏃 Ready to Run"),
        run_benchmark_btn,
        mo.md("_Once you click the button above, the benchmark will execute and results will appear below._")
    ])
    return


@app.cell
def __(np, pd, cudf_module, time, cudf_available):
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
            df2 = cudf_module.DataFrame({
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
    benchmark_functions, cudf_module, cudf_available, pd, mo, np, torch, device
):
    """Run benchmarks"""
    
    benchmark_results = None
    
    # Check if button was clicked (value increments on each click)
    if run_benchmark_btn.value > 0:
        # Show spinner in marimo UI (must use as context manager)
        with mo.status.spinner(subtitle=f"Benchmark running: {10**dataset_size.value:,} rows, {len(operations.value)} ops..."):
            print("=" * 80)
            print(f"🚀 BENCHMARK STARTING")
            print("=" * 80)
            print(f"📊 Dataset: {10**dataset_size.value:,} rows")
            print(f"🔧 Operations: {', '.join(operations.value)}")
            print(f"⚙️  Mode: {mode_toggle.value}")
            print(f"🕐 Estimated time: 10-60 seconds (depending on dataset size)")
            print("=" * 80)
            print("\n⏳ Running benchmark... (watch for progress below)\n")
            
            # Set random seeds for reproducibility
            np.random.seed(42)
            torch.manual_seed(42)
            
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
                
                # Normalize mode - dropdown returns labels, not keys
                _run_cpu = _mode in ['CPU Only', 'CPU vs GPU', 'cpu', 'both']
                _run_gpu = _mode in ['GPU Only', 'CPU vs GPU', 'gpu', 'both']
                
                # Initialize GPU monitoring (for all GPUs)
                _gpu_metrics = {
                    'peak_utilization': 0,
                    'peak_memory_mb': 0,
                    'avg_utilization': 0,
                    'samples': [],
                    'per_gpu': {},  # Track each GPU separately
                    'timeline': []  # Track metrics over time with timestamps
                }
                
                # Record start time for timeline
                import time as time_module
                _benchmark_start_time = time_module.time()
                
                # Try to import GPUtil for GPU monitoring (use unique name to avoid conflicts)
                try:
                    import GPUtil as _GPUtil_bench
                    _gputil_available = True
                except ImportError:
                    _GPUtil_bench = None
                    _gputil_available = False
                
                # Set which GPUs to use - all available GPUs
                _gpu_ids = list(range(num_gpus)) if num_gpus > 0 else [0]
                print(f"💻 Using {len(_gpu_ids)} GPU(s): {_gpu_ids}")
                
                # Convert pandas DataFrame to cuDF once if needed
                _cudf_df = None
                
                if _run_gpu and cudf_available and cudf_module is not None:
                    try:
                        _cudf_df = cudf_module.from_pandas(pandas_df)
                        print(f"✅ Created cuDF DataFrame: {len(_cudf_df):,} rows")
                        
                        # GPU warmup - run a simple operation to initialize GPU
                        # This eliminates first-run overhead from benchmarks
                        _ = _cudf_df[_cudf_df['value'] > 0]
                        print("✅ GPU initialized")
                    except Exception as e:
                        print(f"❌ Failed to create cuDF DataFrame: {e}")
                        import traceback
                        traceback.print_exc()
                        _cudf_df = None
                
                for op in operations.value:
                    if op not in benchmark_functions:
                        continue
                    
                    bench_func = benchmark_functions[op]
                    print(f"\n🔄 Running {op}...")
                    
                    # Record operation start for timeline
                    _op_start_time = time_module.time()
                    
                    # Pandas benchmark (run if mode includes CPU)
                    if _run_cpu:
                        try:
                            pandas_time, result_size = bench_func(pandas_df.copy(), is_gpu=False)
                            print(f"  ✅ Pandas: {pandas_time:.4f}s ({result_size:,} rows)")
                        except Exception as e:
                            print(f"  ❌ Pandas failed: {e}")
                            pandas_time, result_size = None, 0
                    else:
                        pandas_time, result_size = None, 0
                    
                    # cuDF benchmark (run if mode includes GPU AND cuDF available)
                    if _run_gpu and _cudf_df is not None:
                        try:
                            # Capture GPU metrics before operation (all GPUs)
                            if _gputil_available:
                                try:
                                    _gpus_bench = _GPUtil_bench.getGPUs()
                                    _elapsed = time_module.time() - _benchmark_start_time
                                    
                                    for _gpu_bench in _gpus_bench:
                                        if _gpu_bench.id in _gpu_ids:
                                            # Track per-GPU metrics
                                            if _gpu_bench.id not in _gpu_metrics['per_gpu']:
                                                _gpu_metrics['per_gpu'][_gpu_bench.id] = {'peak_util': 0, 'peak_mem': 0}
                                            
                                            _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_util'] = max(
                                                _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_util'], 
                                                _gpu_bench.load * 100
                                            )
                                            _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_mem'] = max(
                                                _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_mem'], 
                                                _gpu_bench.memoryUsed
                                            )
                                            
                                            # Track overall metrics
                                            _gpu_metrics['samples'].append({
                                                'util': _gpu_bench.load * 100,
                                                'mem': _gpu_bench.memoryUsed,
                                                'gpu_id': _gpu_bench.id
                                            })
                                            _gpu_metrics['peak_utilization'] = max(_gpu_metrics['peak_utilization'], _gpu_bench.load * 100)
                                            _gpu_metrics['peak_memory_mb'] = max(_gpu_metrics['peak_memory_mb'], _gpu_bench.memoryUsed)
                                            
                                            # Add to timeline for trend chart
                                            _gpu_metrics['timeline'].append({
                                                'timestamp': _elapsed,
                                                'operation': op,
                                                'phase': 'start',
                                                'gpu_id': _gpu_bench.id,
                                                'util': _gpu_bench.load * 100,
                                                'mem_mb': _gpu_bench.memoryUsed,
                                                'temp': _gpu_bench.temperature if hasattr(_gpu_bench, 'temperature') else None
                                            })
                                except:
                                    pass
                            
                            cudf_time, _ = bench_func(_cudf_df, is_gpu=True)
                            
                            # Capture GPU metrics after operation (all GPUs)
                            if _gputil_available:
                                try:
                                    _gpus_bench = _GPUtil_bench.getGPUs()
                                    _elapsed = time_module.time() - _benchmark_start_time
                                    
                                    for _gpu_bench in _gpus_bench:
                                        if _gpu_bench.id in _gpu_ids:
                                            # Track per-GPU metrics
                                            if _gpu_bench.id not in _gpu_metrics['per_gpu']:
                                                _gpu_metrics['per_gpu'][_gpu_bench.id] = {'peak_util': 0, 'peak_mem': 0}
                                            
                                            _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_util'] = max(
                                                _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_util'], 
                                                _gpu_bench.load * 100
                                            )
                                            _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_mem'] = max(
                                                _gpu_metrics['per_gpu'][_gpu_bench.id]['peak_mem'], 
                                                _gpu_bench.memoryUsed
                                            )
                                            
                                            # Track overall metrics
                                            _gpu_metrics['samples'].append({
                                                'util': _gpu_bench.load * 100,
                                                'mem': _gpu_bench.memoryUsed,
                                                'gpu_id': _gpu_bench.id
                                            })
                                            _gpu_metrics['peak_utilization'] = max(_gpu_metrics['peak_utilization'], _gpu_bench.load * 100)
                                            _gpu_metrics['peak_memory_mb'] = max(_gpu_metrics['peak_memory_mb'], _gpu_bench.memoryUsed)
                                            
                                            # Add to timeline for trend chart
                                            _gpu_metrics['timeline'].append({
                                                'timestamp': _elapsed,
                                                'operation': op,
                                                'phase': 'end',
                                                'gpu_id': _gpu_bench.id,
                                                'util': _gpu_bench.load * 100,
                                                'mem_mb': _gpu_bench.memoryUsed,
                                                'temp': _gpu_bench.temperature if hasattr(_gpu_bench, 'temperature') else None
                                            })
                                except:
                                    pass
                            
                            speedup = pandas_time / cudf_time if (pandas_time and cudf_time) else None
                            print(f"  ✅ cuDF: {cudf_time:.4f}s (speedup: {speedup:.1f}x)" if speedup else f"  ✅ cuDF: {cudf_time:.4f}s")
                        except Exception as e:
                            print(f"  ❌ cuDF failed: {e}")
                            import traceback
                            traceback.print_exc()
                            cudf_time = None
                            speedup = None
                    else:
                        cudf_time = None
                        speedup = None
                    
                    _results['operation'].append(op.capitalize())
                    _results['pandas_time'].append(pandas_time)
                    _results['cudf_time'].append(cudf_time)
                    _results['speedup'].append(speedup)
                    _results['result_size'].append(result_size)
                
                # Calculate average GPU utilization
                if _gpu_metrics['samples']:
                    _gpu_metrics['avg_utilization'] = sum(s['util'] for s in _gpu_metrics['samples']) / len(_gpu_metrics['samples'])
                
                benchmark_results = {
                    'results': _results,
                    'n_rows': _n_rows,
                    'mode': _mode,
                    'success': True,
                    'gpu_metrics': _gpu_metrics if _run_gpu and _gpu_metrics['samples'] else None
                }
                
                # Cleanup GPU memory
                if _cudf_df is not None:
                    try:
                        del _cudf_df
                        torch.cuda.empty_cache()
                        print("✅ Cleaned up GPU memory")
                    except Exception as e:
                        print(f"⚠️  GPU cleanup warning: {e}")
                
                # Final completion message
                print("\n" + "=" * 80)
                print("🎉 BENCHMARK COMPLETE!")
                print("=" * 80)
                print("📊 Results and GPU timeline charts are displayed below.")
                print("=" * 80 + "\n")
            
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
    
    # Build output based on benchmark state
    _output = None
    
@app.cell
def __(mo, run_benchmark_btn):
    """Benchmark Results Section Header (only after benchmark runs)"""
    # Only show this header after user has clicked the run button
    mo.stop(run_benchmark_btn.value == 0, mo.md(""))
    
    mo.md("## 📊 Benchmark Results")
    return


@app.cell
def __(mo):
    """Educational: Understanding GPU Metrics"""
    mo.vstack([
        mo.md("---"),
        mo.md("## 📊 Understanding GPU Metrics"),
        mo.accordion({
            "Click to expand educational content": mo.md("""
### GPU Utilization (%)
**What it means**: Percentage of time GPU cores are actively computing

**Target values:**
- **< 30%**: Underutilized - data transfer overhead or small dataset
- **30-70%**: Moderate - good for mixed workloads
- **> 70%**: Well utilized - GPU is the bottleneck (this is good!)
- **> 95%**: Fully saturated - maximum performance achieved

**Why it matters:**
- Low utilization = you're paying for GPU but not using it
- High utilization = GPU acceleration is working
- Consistent 100% = might need bigger GPU or data batching

### Memory Usage (GB/%)
**What it means**: How much GPU VRAM is allocated

**Why monitor it:**
- **Out of Memory**: Most common GPU error
- **Memory fragmentation**: Can cause OOM even with free memory
- **Batch size tuning**: Larger batches = better utilization but more memory

**VRAM vs System RAM:**
- System RAM: 64-512 GB typical (slow, CPU accessible)
- GPU VRAM: 16-80 GB typical (fast, GPU-only)
- Transfer speed: ~16 GB/s PCIe (bottleneck!)

**Rule of thumb:**
- Use <80% of VRAM to leave headroom
- Monitor peak memory, not average
- cuDF uses ~2-3x data size in memory (intermediate results)

### Temperature (°C)
**What it means**: GPU core temperature

**Normal ranges:**
- **Idle**: 30-50°C
- **Light load**: 50-65°C
- **Heavy load**: 65-80°C
- **Throttling starts**: 80-85°C (GPU slows down to cool)
- **Critical**: 90°C+ (emergency shutdown)

**Why it matters:**
- Hot GPU = thermal throttling = slower performance
- Consistent high temps = check cooling/airflow
- Data center GPUs (A100, L40S) have better cooling than consumer GPUs

**Performance impact:**
- Every 10°C above 65°C = ~5-10% performance loss
- Throttling at 85°C = 15-30% performance loss
            """)
        }),
        mo.md("---")
    ])
    return


@app.cell
def __(benchmark_results, mo, pd, go, cudf_available, run_benchmark_btn):
    """Visualize benchmark results"""
    
    if benchmark_results is None:
        # Don't show anything - status cell handles this
        _output = None
    elif not benchmark_results.get('success', False):
        # Show error message
        error_msg = f"**Benchmark Error**: {benchmark_results['error']}"
        if 'suggestion' in benchmark_results:
            error_msg += f"\n\n{benchmark_results['suggestion']}"
        if 'error_type' in benchmark_results:
            error_msg += f"\n\n*Error type: {benchmark_results['error_type']}*"
        
        _output = mo.callout(
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
        
        # Extract mode flags from benchmark_results
        _mode = benchmark_results['mode']
        _run_cpu = _mode in ['CPU Only', 'CPU vs GPU', 'cpu', 'both']
        _run_gpu = _mode in ['GPU Only', 'CPU vs GPU', 'gpu', 'both']
        
        # Add Pandas times if they were run
        if _run_cpu and any(_results['pandas_time']):
            table_data['Pandas Time (s)'] = [f"{t:.4f}" if t else "N/A" for t in _results['pandas_time']]
        
        # Add cuDF times if they were run
        if _run_gpu and cudf_available and any(_results['cudf_time']):
            table_data['cuDF Time (s)'] = [f"{t:.4f}" if t else "N/A" for t in _results['cudf_time']]
        
        # Add speedup if both were run
        if _run_cpu and _run_gpu and any(_results['speedup']):
            table_data['Speedup'] = [f"{s:.2f}x" if s else "N/A" for s in _results['speedup']]
        
        # Create performance visualization
        fig = go.Figure()
        
        # Add Pandas performance if run
        if _run_cpu and any(_results['pandas_time']):
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
        if _run_gpu and cudf_available and any(_results['cudf_time']):
            cudf_times = [t if t else 0 for t in _results['cudf_time']]
            fig.add_trace(go.Bar(
                name='cuDF (GPU)',
                x=_results['operation'],
                y=cudf_times,
                marker_color='#51cf66',
                text=[f"{t:.3f}s" if t else "N/A" for t in _results['cudf_time']],
                textposition='outside'
            ))
        
        # Determine title suffix based on what actually ran
        if _run_cpu and _run_gpu:
            _title_suffix = ' - CPU vs GPU'
        elif _run_gpu:
            _title_suffix = ' - GPU Only'
        elif _run_cpu:
            _title_suffix = ' - CPU Only'
        else:
            _title_suffix = ''
        
        fig.update_layout(
            title=f"Performance Comparison ({_n_rows:,} rows){_title_suffix}",
            xaxis_title="Operation",
            yaxis_title="Time (seconds)",
            barmode='group',
            height=450,
            hovermode='x unified',
            margin=dict(t=80, b=50, l=50, r=50)  # Extra top margin for bar labels
        )
        
        # Show results - use normalized _run_gpu flag instead of checking mode strings
        if cudf_available and _run_gpu and any(_results['cudf_time']):
            # Speedup chart (only with cuDF)
            fig_speedup = go.Figure()
            
            fig_speedup.add_trace(go.Bar(
                x=_results['operation'],
                y=_results['speedup'],
                marker_color='#4c6ef5',
                text=[f"{s:.1f}x" if s is not None else "N/A" for s in _results['speedup']],
                textposition='outside'
            ))
            
            fig_speedup.update_layout(
                title="GPU Speedup Factor",
                xaxis_title="Operation",
                yaxis_title="Speedup (higher is better)",
                height=450,
                margin=dict(t=80, b=50, l=50, r=50)  # Extra top margin for bar labels
            )
            
            # Calculate overall statistics (filter out None values)
            valid_speedups = [s for s in _results['speedup'] if s is not None]
            if valid_speedups:
                avg_speedup = sum(valid_speedups) / len(valid_speedups)
                max_speedup = max(valid_speedups)
                min_speedup = min(valid_speedups)
            else:
                avg_speedup = 0
                max_speedup = 0
                min_speedup = 0
            
            # Determine callout kind and message based on performance and GPU utilization
            _gpu_util = benchmark_results.get('gpu_metrics', {}).get('peak_utilization', 0)
            
            if avg_speedup >= 5:
                _perf_kind = "success"
                if _gpu_util >= 90:
                    _perf_note = "🚀 GPU is crushing it at full throttle! This is the sweet spot for GPU acceleration."
                elif _gpu_util >= 70:
                    _perf_note = "🚀 GPU is crushing it! Try increasing to 100M rows to hit 100% utilization."
                else:
                    _perf_note = "🚀 Great speedup! GPU still has headroom - try larger datasets (50M-100M rows)."
            elif avg_speedup >= 2:
                _perf_kind = "success"
                if _gpu_util >= 90:
                    _perf_note = "✅ GPU is significantly faster at full utilization!"
                elif _gpu_util >= 50:
                    _perf_note = "✅ GPU is significantly faster! Try 10M-100M rows to push GPU to 100% utilization."
                else:
                    _perf_note = f"✅ GPU is faster, but only using {_gpu_util:.0f}% - increase dataset size for more speedup."
            elif avg_speedup >= 1:
                _perf_kind = "info"
                _perf_note = "💡 GPU shows modest gains. Increase dataset size (10M+ rows) for dramatic speedup."
            else:
                _perf_kind = "warn"
                _perf_note = "⚠️ CPU is faster on this dataset size. GPU excels with 10M+ rows due to parallelism overhead."
            
            # Build GPU metrics string if available
            _gpu_metrics_str = ""
            if benchmark_results.get('gpu_metrics'):
                _gm = benchmark_results['gpu_metrics']
                _gpu_metrics_str = f"""
                    
                    **GPU Utilization During Benchmark**:
                    - Peak GPU Utilization: **{_gm['peak_utilization']:.1f}%** (across all GPUs)
                    - Average GPU Utilization: **{_gm['avg_utilization']:.1f}%**
                    - Peak GPU Memory: **{_gm['peak_memory_mb']:.0f} MB**
                """
                
                # Add per-GPU breakdown if multiple GPUs
                if len(_gm.get('per_gpu', {})) > 1:
                    _gpu_metrics_str += "\n                    \n                    **Per-GPU Breakdown**:\n"
                    for gpu_id in sorted(_gm['per_gpu'].keys()):
                        gpu_data = _gm['per_gpu'][gpu_id]
                        _gpu_metrics_str += f"                    - GPU {gpu_id}: Peak {gpu_data['peak_util']:.1f}% util, {gpu_data['peak_mem']:.0f} MB\n"
            
            # Create GPU timeline charts if we have timeline data
            _gpu_timeline_charts = []
            if benchmark_results.get('gpu_metrics', {}).get('timeline'):
                _timeline = benchmark_results['gpu_metrics']['timeline']
                
                if _timeline:
                    # GPU Utilization over time
                    fig_gpu_util = go.Figure()
                    for gpu_id in sorted(set(t['gpu_id'] for t in _timeline)):
                        gpu_data = [t for t in _timeline if t['gpu_id'] == gpu_id]
                        fig_gpu_util.add_trace(go.Scatter(
                            x=[t['timestamp'] for t in gpu_data],
                            y=[t['util'] for t in gpu_data],
                            mode='lines+markers',
                            name=f'GPU {gpu_id}',
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
                    
                    fig_gpu_util.update_layout(
                        title="GPU Utilization Over Time",
                        xaxis_title="Time (seconds)",
                        yaxis_title="GPU Utilization (%)",
                        yaxis_range=[0, 105],
                        height=350,
                        margin=dict(t=60, b=60, l=60, r=40),
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    # GPU Memory over time
                    fig_gpu_mem = go.Figure()
                    for gpu_id in sorted(set(t['gpu_id'] for t in _timeline)):
                        gpu_data = [t for t in _timeline if t['gpu_id'] == gpu_id]
                        fig_gpu_mem.add_trace(go.Scatter(
                            x=[t['timestamp'] for t in gpu_data],
                            y=[t['mem_mb'] for t in gpu_data],
                            mode='lines+markers',
                            name=f'GPU {gpu_id}',
                            line=dict(width=2),
                            marker=dict(size=6)
                        ))
                    
                    fig_gpu_mem.update_layout(
                        title="GPU Memory Usage Over Time",
                        xaxis_title="Time (seconds)",
                        yaxis_title="GPU Memory (MB)",
                        height=350,
                        margin=dict(t=60, b=60, l=60, r=40),
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    _gpu_timeline_charts.extend([
                        mo.md("### 📈 GPU Performance Timeline"),
                        mo.ui.plotly(fig_gpu_util),
                        mo.ui.plotly(fig_gpu_mem)
                    ])
                    
                    # Add temperature chart if available
                    if any(t.get('temp') is not None for t in _timeline):
                        fig_gpu_temp = go.Figure()
                        for gpu_id in sorted(set(t['gpu_id'] for t in _timeline)):
                            gpu_data = [t for t in _timeline if t['gpu_id'] == gpu_id and t.get('temp') is not None]
                            if gpu_data:
                                fig_gpu_temp.add_trace(go.Scatter(
                                    x=[t['timestamp'] for t in gpu_data],
                                    y=[t['temp'] for t in gpu_data],
                                    mode='lines+markers',
                                    name=f'GPU {gpu_id}',
                                    line=dict(width=2),
                                    marker=dict(size=6)
                                ))
                        
                        fig_gpu_temp.update_layout(
                            title="GPU Temperature Over Time",
                            xaxis_title="Time (seconds)",
                            yaxis_title="Temperature (°C)",
                            height=350,
                            margin=dict(t=60, b=60, l=60, r=40),
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        _gpu_timeline_charts.append(mo.ui.plotly(fig_gpu_temp))
            
            _output = mo.vstack([
                mo.md("### ✅ Benchmark Complete!"),
                mo.ui.table(table_data, label="Performance Results"),
                mo.md("### 📊 Execution Time Comparison"),
                mo.ui.plotly(fig),
                mo.md("### ⚡ Speedup Analysis"),
                mo.ui.plotly(fig_speedup),
                *_gpu_timeline_charts,  # Add GPU timeline charts here
                mo.callout(
                    mo.md(f"""
                    **Performance Summary**:
                    - Average Speedup: **{avg_speedup:.1f}x**
                    - Best Speedup: **{max_speedup:.1f}x** ({_results['operation'][_results['speedup'].index(max_speedup)]})
                    - Minimum Speedup: **{min_speedup:.1f}x**
                    - Dataset Size: **{_n_rows:,}** rows
                    {_gpu_metrics_str}
                    
                    {_perf_note}
                    """),
                    kind=_perf_kind
                )
            ])
        else:
            # CPU-only mode
            _output = mo.vstack([
                mo.md("### ✅ Benchmark Complete (CPU-Only Mode)!"),
                mo.callout(
                    mo.md(f"""
                    ⚠️ **Running in CPU-only mode** (cuDF not available or failed)
                    
                    To install cuDF:
                    ```bash
                    pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
                    ```
                    
                    After installation, restart the notebook to enable GPU benchmarks.
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
                    - Total Time: **{sum(t for t in _results['pandas_time'] if t is not None):.2f}s**
                    - Average Time per Operation: **{(sum(t for t in _results['pandas_time'] if t is not None)/len([t for t in _results['pandas_time'] if t is not None])) if [t for t in _results['pandas_time'] if t is not None] else 0:.3f}s**
                    
                    💡 With cuDF (GPU), these operations could be **10-100x faster**!
                    """),
                    kind="info"
                )
            ])
    
    # Return the output
    _output


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

