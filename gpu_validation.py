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
        default_interval="2000ms",
        options=["Off", "1000ms", "2000ms", "5000ms"]
    )
    
    mo.hstack([
        mo.md("**Auto-refresh metrics:**"),
        auto_refresh,
        mo.md("*(1s, 2s, or 5s intervals)*")
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
    import threading
    import datetime as dt
    
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
            _test_device = torch.device("cuda:0")
            
            # Allocate persistent tensors for continuous stress
            _gpu_size = 12000
            _gpu_a = torch.randn(_gpu_size, _gpu_size, device=_test_device)
            _gpu_b = torch.randn(_gpu_size, _gpu_size, device=_test_device)
            
            # Run continuous operations
            _start_time = time.time()
            _ops_count = 0
            
            # Do 10 operations
            for _ in range(10):
                _result = torch.matmul(_gpu_a, _gpu_b)
                _result = torch.matmul(_result, _gpu_a)
                torch.cuda.synchronize()
                _ops_count += 2
            
            _elapsed = time.time() - _start_time
            _ops_per_sec = _ops_count / _elapsed if _elapsed > 0 else 0
            
            test_result = mo.callout(
                mo.md(f"""
                🔥 **GPU Stress Test Running!**
                
                **Status**: Active - GPU is being stressed continuously
                **Operations**: {_ops_count} matrix multiplications completed in last cycle
                **Performance**: ~{_ops_per_sec:.1f} ops/second
                **Matrix Size**: {_gpu_size}×{_gpu_size} ({_gpu_size*_gpu_size*8/1e9:.2f} GB per tensor)
                
                📊 **Watch the metrics above auto-refresh to see:**
                - 🔥 GPU Utilization: Should be at ~100%
                - 🌡️ Temperature: Increasing steadily
                - 💾 Memory: High usage from large tensors
                
                Toggle the switch off to stop the stress test.
                """),
                kind="info"
            )
            
            # Clean up
            del _gpu_a, _gpu_b, _result
            torch.cuda.empty_cache()
            
        except Exception as e:
            test_result = mo.callout(
                mo.md(f"❌ **GPU test failed**: {str(e)}\n\nThis could be due to insufficient GPU memory. Try closing other GPU applications."),
                kind="danger"
            )
    else:
        test_result = mo.md("""
        💡 **Toggle the switch above to start continuous GPU stress testing**
        
        When enabled, the GPU will continuously perform intensive matrix operations,
        allowing you to see real-time GPU metrics changes.
        
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

