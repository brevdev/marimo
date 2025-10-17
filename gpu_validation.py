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
    gpu_libs_error = None
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
    # Auto-refresh at 2s - smooth CSS transitions make it feel seamless
    # Must display for it to work, but hide it with CSS
    auto_refresh = mo.ui.refresh(default_interval="2s")
    
    # Display but make invisible
    mo.Html(f"""
    <div style="height: 0; overflow: hidden; opacity: 0; pointer-events: none;">
        {auto_refresh}
    </div>
    """)
    return auto_refresh,


@app.cell
def __(GPUtil, auto_refresh, mo, time):
    # Trigger on auto-refresh
    _refresh_trigger = auto_refresh.value
    _update_time = time.strftime("%H:%M:%S")
    
    try:
        current_gpus = GPUtil.getGPUs()
        
        if not current_gpus:
            gpu_metrics = mo.callout(
                mo.md("No GPUs available to monitor"),
                kind="warn"
            )
        else:
            # Create modern card-based display with smooth CSS transitions
            gpu_cards_html = []
            
            for gpu in current_gpus:
                util_pct = round(gpu.load * 100, 1)
                mem_pct = round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1)
                temp = gpu.temperature
                power_info = ""
                if hasattr(gpu, 'powerDraw') and gpu.powerDraw:
                    power_info = f"<div style='font-size: 0.8em; color: #888; margin-top: 4px;'>Power: {gpu.powerDraw}W / {gpu.powerLimit}W</div>"
                
                card_html = f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 20px; background: #f9f9f9; min-height: 200px;">
                    <h3 style="margin: 0 0 15px 0; font-size: 1.1em;">GPU {gpu.id}: {gpu.name}</h3>
                    
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                        <!-- Utilization Card -->
                        <div style="background: white; padding: 15px; border-radius: 6px; border-left: 4px solid #4CAF50; min-height: 100px;">
                            <div style="font-size: 0.85em; color: #666; margin-bottom: 5px; font-weight: 500;">Utilization</div>
                            <div style="font-size: 1.8em; font-weight: bold; color: #333; margin-bottom: 8px;">{util_pct}%</div>
                            <div style="height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); width: {util_pct}%; transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                            </div>
                        </div>
                        
                        <!-- Memory Card -->
                        <div style="background: white; padding: 15px; border-radius: 6px; border-left: 4px solid #2196F3; min-height: 100px;">
                            <div style="font-size: 0.85em; color: #666; margin-bottom: 5px; font-weight: 500;">Memory</div>
                            <div style="font-size: 1.8em; font-weight: bold; color: #333; margin-bottom: 4px;">{mem_pct}%</div>
                            <div style="font-size: 0.75em; color: #888; margin-bottom: 8px;">{gpu.memoryUsed:.0f} MB / {gpu.memoryTotal:.0f} MB</div>
                            <div style="height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                                <div style="height: 100%; background: linear-gradient(90deg, #2196F3 0%, #1976D2 100%); width: {mem_pct}%; transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                            </div>
                        </div>
                        
                        <!-- Temperature Card -->
                        <div style="background: white; padding: 15px; border-radius: 6px; border-left: 4px solid #FF9800; min-height: 100px;">
                            <div style="font-size: 0.85em; color: #666; margin-bottom: 5px; font-weight: 500;">Temperature</div>
                            <div style="font-size: 1.8em; font-weight: bold; color: #333; margin-bottom: 4px;">{temp}°C</div>
                            {power_info}
                        </div>
                    </div>
                </div>
                """
                gpu_cards_html.append(card_html)
            
            # Combine all GPU cards with update time
            all_cards = "\n".join(gpu_cards_html)
            
            gpu_metrics = mo.Html(f"""
            <div style="position: relative;">
                <div style="text-align: right; color: #888; font-size: 0.85em; margin-bottom: 10px; font-family: monospace;">
                    ⏱️ Last updated: {_update_time}
                </div>
                <div style="display: grid; gap: 20px; animation: fadeIn 0.3s ease-in;">
                    {all_cards}
                </div>
            </div>
            
            <style>
                @keyframes fadeIn {{
                    from {{ opacity: 0.7; transform: translateY(-5px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
            </style>
            """)
            
    except Exception as e:
        gpu_metrics = mo.callout(
            mo.md(f"Error reading GPU metrics: {str(e)}"),
            kind="danger"
        )
    
    gpu_metrics
    return gpu_metrics,


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
def __(mo, psutil, stress_test_running, subprocess):
    import shutil
    import os
    
    _is_running = stress_test_running.value
    
    # Check multiple locations for gpu-burn
    _gpu_burn_path = shutil.which("gpu_burn")  # Check PATH first
    
    if not _gpu_burn_path:
        # Check if compiled in home directory
        _home_gpu_burn = os.path.expanduser("~/gpu-burn/gpu_burn")
        if os.path.exists(_home_gpu_burn):
            _gpu_burn_path = _home_gpu_burn
    
    # Try to install gpu-burn if not found
    if not _gpu_burn_path:
        _install_msg = []
        _install_msg.append("🔧 **gpu-burn not found - attempting to install...**\n")
        
        try:
            # Skip apt-get (often not in repos), go straight to source compile
            _install_msg.append("📦 Compiling gpu-burn from source...")
            
            # Compile from source
            _home_dir = os.path.expanduser("~")
            _gpu_burn_dir = os.path.join(_home_dir, "gpu-burn")
            
            if not os.path.exists(_gpu_burn_dir):
                subprocess.run(
                    ["git", "clone", "https://github.com/wilicc/gpu-burn.git", _gpu_burn_dir],
                    check=True,
                    timeout=30,
                    cwd=_home_dir
                )
            
            subprocess.run(["make"], check=True, timeout=60, cwd=_gpu_burn_dir)
            
            # Set path to compiled binary
            _gpu_burn_path = os.path.join(_gpu_burn_dir, "gpu_burn")
            
            if os.path.exists(_gpu_burn_path):
                _install_msg.append(f"✅ Successfully compiled gpu-burn from source!")
                _install_msg.append(f"📍 Binary location: {_gpu_burn_path}")
            else:
                _install_msg.append(f"❌ Compilation succeeded but binary not found at {_gpu_burn_path}")
                _gpu_burn_path = None
                
        except Exception as e:
            _install_msg.append(f"❌ Installation failed: {str(e)}")
            _gpu_burn_path = None
        
        if not _gpu_burn_path:
            test_result = mo.callout(
                mo.md("\n".join(_install_msg) + "\n\n**Manual installation**: Compile from [github.com/wilicc/gpu-burn](https://github.com/wilicc/gpu-burn)"),
                kind="danger"
            )
        else:
            test_result = mo.callout(
                mo.md("\n".join(_install_msg) + "\n\n**Ready!** Toggle the switch to start stress testing."),
                kind="success"
            )
    elif _is_running:
        try:
            # Check if there's already a gpu-burn process running (using psutil from cell-2)
            _gpu_burn_running = False
            _gpu_burn_pid = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and 'gpu_burn' in ' '.join(proc.info['cmdline']):
                        _gpu_burn_running = True
                        _gpu_burn_pid = proc.info['pid']
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Start gpu-burn if not already running
            if not _gpu_burn_running:
                _gpu_burn_dir = os.path.dirname(_gpu_burn_path)
                
                # Run gpu-burn in background for long duration (1 hour)
                # We'll kill it when switch is toggled off
                _process = subprocess.Popen(
                    [_gpu_burn_path, "3600"],  # Run for 1 hour
                    cwd=_gpu_burn_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                _gpu_burn_pid = _process.pid
                _status_msg = f"✅ Started gpu-burn (PID: {_gpu_burn_pid})"
            else:
                _status_msg = f"🔥 gpu-burn already running (PID: {_gpu_burn_pid})"
            
            test_result = mo.callout(
                mo.md(f"""
                🔥 **gpu-burn Stress Test ACTIVE!**
                
                **Status**: {_status_msg}  
                **Tool**: Industry-standard [gpu-burn](https://github.com/wilicc/gpu-burn)  
                **Binary**: `{_gpu_burn_path}`  
                **Process ID**: {_gpu_burn_pid}  
                **Intensity**: MAXIMUM (95% GPU memory + double precision)
                
                📊 **Metrics are now updating in real-time!**  
                ⚠️ **gpu-burn runs in background** - metrics will refresh while it runs
                
                Watch the metrics above auto-refresh to see 100% utilization!  
                Toggle off to stop gpu-burn.
                """),
                kind="info"
            )
            
        except Exception as e:
            test_result = mo.callout(
                mo.md(f"❌ **gpu-burn test failed**: {str(e)}\n\n**Path tried**: `{_gpu_burn_path}`"),
                kind="danger"
            )
    else:
        # Kill any running gpu-burn processes when switch is off (using psutil from cell-2)
        _killed_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and 'gpu_burn' in ' '.join(proc.info['cmdline']):
                    proc.kill()
                    _killed_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if _killed_processes:
            test_result = mo.callout(
                mo.md(f"🛑 **Stopped gpu-burn** (killed PID: {', '.join(map(str, _killed_processes))})\n\nGPU will cool down now. Metrics will return to idle state."),
                kind="success"
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
            
            **Runs in background so metrics update in real-time!**
            
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

