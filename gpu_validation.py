import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        """
        # 🚀 NVIDIA GPU Validation & Monitoring

        This notebook validates your GPU setup and provides real-time monitoring of GPU metrics.
        Perfect for ensuring your GPU environment is properly configured before running experiments.
        """
    )
    return


@app.cell
def __():
    import subprocess
    import sys
    import platform
    
    # GPU monitoring libraries
    try:
        import pynvml
        import GPUtil
        import psutil
        gpu_libs_available = True
    except ImportError as e:
        gpu_libs_available = False
        gpu_libs_error = str(e)
    
    # ML frameworks
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    # Visualization
    import pandas as pd
    import altair as alt
    from datetime import datetime
    import time
    return (
        GPUtil,
        alt,
        datetime,
        gpu_libs_available,
        gpu_libs_error,
        pd,
        platform,
        psutil,
        pynvml,
        subprocess,
        sys,
        time,
        torch,
        torch_available,
    )


@app.cell
def __(mo):
    mo.md("""## 🔍 System Information""")
    return


@app.cell
def __(mo, platform, sys):
    system_info = mo.md(
        f"""
        - **OS**: {platform.system()} {platform.release()}
        - **Python Version**: {sys.version.split()[0]}
        - **Architecture**: {platform.machine()}
        """
    )
    system_info
    return system_info,


@app.cell
def __(mo):
    mo.md("""## 🎮 GPU Detection""")
    return


@app.cell
def __(GPUtil, gpu_libs_available, mo, pynvml, torch, torch_available):
    if not gpu_libs_available:
        gpu_status = mo.callout(
            mo.md("⚠️ **GPU monitoring libraries not available**"),
            kind="warn"
        )
        device_count = 0
        detected_gpus = []
    else:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                gpu_status = mo.callout(
                    mo.md("❌ **No NVIDIA GPUs detected**"),
                    kind="danger"
                )
                detected_gpus = []
            else:
                detected_gpus = GPUtil.getGPUs()
                gpu_info_lines = [f"✅ **{device_count} NVIDIA GPU(s) detected**\n"]
                
                for idx, detected_gpu in enumerate(detected_gpus):
                    gpu_info_lines.append(f"\n### GPU {idx}: {detected_gpu.name}")
                    gpu_info_lines.append(f"- **Memory**: {detected_gpu.memoryTotal} MB")
                    gpu_info_lines.append(f"- **Driver Version**: {detected_gpu.driver}")
                    gpu_info_lines.append(f"- **UUID**: {detected_gpu.uuid}")
                
                # Check PyTorch CUDA availability
                if torch_available:
                    gpu_info_lines.append(f"\n### PyTorch CUDA")
                    gpu_info_lines.append(f"- **CUDA Available**: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        gpu_info_lines.append(f"- **CUDA Version**: {torch.version.cuda}")
                        gpu_info_lines.append(f"- **PyTorch GPU Count**: {torch.cuda.device_count()}")
                        gpu_info_lines.append(f"- **Current Device**: {torch.cuda.current_device()}")
                        gpu_info_lines.append(f"- **Device Name**: {torch.cuda.get_device_name(0)}")
                
                gpu_status = mo.callout(
                    mo.md("\n".join(gpu_info_lines)),
                    kind="success"
                )
        except Exception as e:
            gpu_status = mo.callout(
                mo.md(f"❌ **Error accessing GPU**: {str(e)}"),
                kind="danger"
            )
            detected_gpus = []
    
    gpu_status
    return detected_gpus, device_count, gpu_status


@app.cell
def __(mo):
    mo.md("""## 📊 Real-time GPU Metrics""")
    return


@app.cell
def __(mo):
    # Auto-refresh controls
    auto_refresh = mo.ui.refresh(
        default_interval="2s",
        options=["1s", "2s", "5s", "10s"]
    )
    
    mo.hstack([
        mo.md("**Auto-refresh metrics:**"),
        auto_refresh,
        mo.md("*(Choose refresh interval - uncheck to disable)*")
    ])
    return auto_refresh,


@app.cell
def __(GPUtil, alt, auto_refresh, mo, pd):
    # Trigger on auto-refresh
    _refresh_trigger = auto_refresh.value
    
    try:
        current_gpus = GPUtil.getGPUs()
        
        if not current_gpus:
            gpu_metrics = mo.callout(
                mo.md("No GPUs available to monitor"),
                kind="warn"
            )
        else:
            # Create metrics for each GPU
            metrics_data = []
            for current_gpu in current_gpus:
                metrics_data.append({
                    'GPU': f"GPU {current_gpu.id}",
                    'Metric': 'Utilization',
                    'Value': current_gpu.load * 100,
                    'Unit': '%'
                })
                metrics_data.append({
                    'GPU': f"GPU {current_gpu.id}",
                    'Metric': 'Memory Used',
                    'Value': (current_gpu.memoryUsed / current_gpu.memoryTotal) * 100,
                    'Unit': '%'
                })
                metrics_data.append({
                    'GPU': f"GPU {current_gpu.id}",
                    'Metric': 'Temperature',
                    'Value': current_gpu.temperature,
                    'Unit': '°C'
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create bar chart
            metrics_chart = alt.Chart(df_metrics).mark_bar().encode(
                x=alt.X('Value:Q', title='Value'),
                y=alt.Y('Metric:N', title=''),
                color=alt.Color('GPU:N', legend=alt.Legend(title="GPU")),
                row=alt.Row('GPU:N', title='')
            ).properties(
                width=600,
                height=100,
                title='GPU Metrics Overview'
            )
            
            # Create detailed info cards
            gpu_cards = []
            for current_gpu in current_gpus:
                gpu_card = mo.md(f"""
                ### GPU {current_gpu.id}: {current_gpu.name}
                
                **Utilization**: {current_gpu.load * 100:.1f}%  
                **Memory**: {current_gpu.memoryUsed:.0f} MB / {current_gpu.memoryTotal:.0f} MB ({(current_gpu.memoryUsed/current_gpu.memoryTotal)*100:.1f}%)  
                **Temperature**: {current_gpu.temperature}°C  
                **Power Draw**: {getattr(current_gpu, 'powerDraw', 'N/A')} W / {getattr(current_gpu, 'powerLimit', 'N/A')} W
                """)
                gpu_cards.append(gpu_card)
            
            gpu_metrics = mo.vstack([
                mo.ui.altair_chart(metrics_chart),
                mo.md("### Detailed Metrics"),
                mo.hstack(gpu_cards)
            ])
    except Exception as e:
        gpu_metrics = mo.callout(
            mo.md(f"Error reading GPU metrics: {str(e)}"),
            kind="danger"
        )
    
    gpu_metrics
    return gpu_metrics, metrics_chart


@app.cell
def __(mo):
    mo.md("""## 🧪 GPU Compute Stress Test""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        Run a continuous GPU stress test to see your GPU in action! Watch the metrics above update in real-time.
        
        **Test Details:**
        - Continuous matrix multiplications until stopped
        - Keeps GPU at high utilization
        - Auto-refreshing metrics show real-time GPU activity
        - Watch utilization, temperature, and memory increase!
        """
    )
    return


@app.cell
def __(mo):
    # Test controls - toggle button
    stress_test_running = mo.ui.switch(label="🔥 GPU Stress Test")
    
    mo.hstack([
        stress_test_running,
        mo.md("Toggle on to start continuous GPU stress test")
    ])
    return stress_test_running,


@app.cell
def __(mo, stress_test_running, time, torch, torch_available):
    _is_running = stress_test_running.value
    
    if not torch_available:
        test_result = mo.callout(
            mo.md("❌ PyTorch not available. Cannot run GPU test."),
            kind="danger"
        )
    elif not torch.cuda.is_available():
        test_result = mo.callout(
            mo.md("❌ CUDA not available. Cannot run GPU test."),
            kind="danger"
        )
    elif _is_running:
        try:
            # Get number of GPUs
            _num_gpus = torch.cuda.device_count()
            _gpu_results = []
            
            # Stress ALL GPUs simultaneously
            for _gpu_id in range(_num_gpus):
                _device = torch.device(f"cuda:{_gpu_id}")
                
                # Use maximum memory for intensive stress (similar to gpu-burn)
                # Allocate multiple large tensors to max out memory and compute
                _gpu_size = 14000  # Larger matrices = more stress
                
                # Allocate 3 tensors per GPU to maximize memory usage
                _tensors = []
                for _i in range(3):
                    _tensors.append(torch.randn(_gpu_size, _gpu_size, device=_device, dtype=torch.float32))
                
                # Run intensive operations (similar to gpu-burn approach)
                _start_time = time.time()
                _ops_count = 0
                
                # Do 20 matrix multiplications to really push the GPU
                for _iter in range(20):
                    # Chain operations to keep GPU busy
                    _temp = torch.matmul(_tensors[0], _tensors[1])
                    _temp = torch.matmul(_temp, _tensors[2])
                    _temp = torch.matmul(_temp, _tensors[0])
                    _ops_count += 3
                
                torch.cuda.synchronize(_device)
                _elapsed = time.time() - _start_time
                
                _ops_per_sec = _ops_count / _elapsed if _elapsed > 0 else 0
                _mem_allocated = torch.cuda.memory_allocated(_gpu_id) / 1e9
                _mem_reserved = torch.cuda.memory_reserved(_gpu_id) / 1e9
                
                _gpu_results.append(
                    f"**GPU {_gpu_id}**: {_ops_count} ops in {_elapsed:.2f}s (~{_ops_per_sec:.1f} ops/s) | Mem: {_mem_allocated:.2f}GB allocated"
                )
                
                # Clean up
                del _tensors, _temp
                torch.cuda.empty_cache()
            
            _result_text = "\n".join([
                "🔥 **Multi-GPU Stress Test Running!**",
                "",
                f"**GPUs Active**: {_num_gpus}",
                f"**Matrix Size**: {_gpu_size}×{_gpu_size} per tensor",
                f"**Tensors per GPU**: 3 (max memory stress)",
                f"**Operations**: 60 chained matrix multiplications per cycle",
                "",
                "📊 **Per-GPU Performance:**",
                *_gpu_results,
                "",
                "🌡️ **Watch the metrics above auto-refresh to see:**",
                "- 🔥 All GPUs at ~100% utilization",
                "- 🌡️ Temperature climbing on all GPUs",
                "- 💾 High memory usage across all GPUs",
                "",
                "⚠️ **This is an intensive stress test similar to gpu-burn**",
                "Toggle off to stop stressing all GPUs."
            ])
            
            test_result = mo.callout(
                mo.md(_result_text),
                kind="info"
            )
            
        except Exception as e:
            test_result = mo.callout(
                mo.md(f"❌ **GPU test failed**: {str(e)}\n\nThis could be due to insufficient GPU memory. The test uses large tensors to maximize stress."),
                kind="danger"
            )
    else:
        test_result = mo.md("""
        💡 **Toggle the switch above to start intensive multi-GPU stress testing**
        
        This test is designed to push ALL GPUs to maximum load (similar to gpu-burn):
        - Uses all available GPUs simultaneously
        - Allocates large tensors to maximize memory usage
        - Performs intensive matrix multiplications continuously
        - Keeps GPUs at ~100% utilization
        
        Make sure auto-refresh is enabled above to see live updates!
        """)
    
    test_result
    return test_result,


@app.cell
def __(mo):
    mo.md("""## 🛠️ nvidia-smi Output""")
    return


@app.cell
def __(mo, subprocess):
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], text=True)
        nvidia_smi_display = mo.accordion({
            "📋 Click to view full nvidia-smi output": mo.md(f"""
```text
{nvidia_smi_output}
```
            """)
        })
    except FileNotFoundError:
        nvidia_smi_display = mo.callout(
            mo.md("❌ nvidia-smi command not found. NVIDIA drivers may not be installed."),
            kind="danger"
        )
    except Exception as e:
        nvidia_smi_display = mo.callout(
            mo.md(f"❌ Error running nvidia-smi: {str(e)}"),
            kind="danger"
        )
    
    nvidia_smi_display
    return nvidia_smi_display, nvidia_smi_output


@app.cell
def __(mo):
    mo.md(
        """
        ## 📝 Summary

        This notebook provides a comprehensive overview of your GPU setup. Use it to:
        - Verify GPU availability before running experiments
        - Monitor GPU utilization during training
        - Troubleshoot GPU-related issues
        - Benchmark GPU performance

        **Next Steps:**
        - Explore the marimo examples in the `marimo-examples` directory
        - Create your own GPU-accelerated notebooks
        - Monitor your GPU usage during model training

        Happy experimenting! 🚀
        """
    )
    return


if __name__ == "__main__":
    app.run()

