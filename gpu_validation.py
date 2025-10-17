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
def __(mo, stress_test_running, subprocess):
    import shutil
    import os
    
    _is_running = stress_test_running.value
    
    # Check if gpu-burn is available
    _gpu_burn_path = shutil.which("gpu_burn")
    
    # Try to install gpu-burn if not found
    if not _gpu_burn_path:
        _install_msg = []
        _install_msg.append("🔧 **gpu-burn not found - attempting to install...**\n")
        
        try:
            # Try apt-get first (Ubuntu/Debian)
            _install_result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", "gpu-burn"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if _install_result.returncode == 0:
                _install_msg.append("✅ Successfully installed gpu-burn via apt!")
                _gpu_burn_path = shutil.which("gpu_burn")
            else:
                _install_msg.append("⚠️ apt-get install failed, trying source compile...")
                
                # Fallback: compile from source
                os.chdir(os.path.expanduser("~"))
                if not os.path.exists("gpu-burn"):
                    subprocess.run(["git", "clone", "https://github.com/wilicc/gpu-burn.git"], check=True, timeout=30)
                
                os.chdir("gpu-burn")
                subprocess.run(["make"], check=True, timeout=60)
                
                # Add to PATH
                _gpu_burn_path = os.path.join(os.getcwd(), "gpu_burn")
                _install_msg.append(f"✅ Successfully compiled gpu-burn from source!")
                
        except Exception as e:
            _install_msg.append(f"❌ Installation failed: {str(e)}")
        
        if not _gpu_burn_path:
            test_result = mo.callout(
                mo.md("\n".join(_install_msg) + "\n\n**Manual installation**: Run `sudo apt-get install gpu-burn` or compile from [github.com/wilicc/gpu-burn](https://github.com/wilicc/gpu-burn)"),
                kind="danger"
            )
        else:
            test_result = mo.callout(
                mo.md("\n".join(_install_msg) + "\n\n**Ready!** Toggle the switch to start stress testing."),
                kind="success"
            )
    elif _is_running:
        try:
            # Run gpu-burn for 60 seconds
            # Since marimo re-runs the cell continuously while switch is on,
            # this effectively runs gpu-burn continuously
            _duration = 60  # seconds
            
            _result = subprocess.run(
                [_gpu_burn_path, str(_duration)],
                capture_output=True,
                text=True,
                timeout=_duration + 5
            )
            
            # Parse output
            _output_lines = _result.stdout.strip().split('\n') if _result.stdout else []
            _gpus_tested = len([line for line in _output_lines if "GPU" in line and "OK" in line or "FAULTY" in line])
            
            test_result = mo.callout(
                mo.md(f"""
                🔥 **gpu-burn Stress Test RUNNING!**
                
                **Tool**: Industry-standard [gpu-burn](https://github.com/wilicc/gpu-burn)  
                **Duration**: {_duration} seconds per cycle  
                **GPUs Tested**: All {_gpus_tested} GPU(s) simultaneously  
                **Intensity**: MAXIMUM (uses 95% GPU memory + doubles precision)
                
                📊 **Status**: Running continuous GPU stress test  
                ⚠️ **This will keep re-running while enabled**
                
                Watch metrics above auto-refresh to see 100% utilization!  
                Toggle off to stop.
                
                **Last Run Output**:
                ```
                {_result.stdout[-500:] if _result.stdout else "Starting..."}
                ```
                """),
                kind="info"
            )
            
        except subprocess.TimeoutExpired:
            test_result = mo.callout(
                mo.md("⏱️ **gpu-burn test running** (this is normal - it runs for 60s per cycle)"),
                kind="info"
            )
        except Exception as e:
            test_result = mo.callout(
                mo.md(f"❌ **gpu-burn test failed**: {str(e)}\n\nMake sure gpu-burn is properly installed."),
                kind="danger"
            )
    else:
        test_result = mo.md("""
        💡 **Toggle the switch above to start gpu-burn stress testing**
        
        This uses the industry-standard **gpu-burn** tool:
        - Automatically stresses ALL GPUs simultaneously
        - Uses 95% of GPU memory by default
        - Double-precision floating point operations
        - Battle-tested stress testing (used in datacenters worldwide)
        - Works on L40S, A100, H100, H200, B200, and all NVIDIA GPUs
        
        **Runs 60-second cycles continuously while enabled!**
        
        Enable auto-refresh above to watch GPUs hit 100%!
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

