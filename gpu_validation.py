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
            _install_msg.append("📦 Checking prerequisites...")
            
            # Check if build tools (gcc, make) are installed
            _has_gcc = subprocess.run(["which", "gcc"], capture_output=True).returncode == 0
            _has_make = subprocess.run(["which", "make"], capture_output=True).returncode == 0
            
            if not _has_gcc or not _has_make:
                _install_msg.append("⚙️ Installing build tools (build-essential)...")
                try:
                    subprocess.run(["sudo", "apt-get", "update"], check=True, timeout=30, capture_output=True)
                    subprocess.run(
                        ["sudo", "apt-get", "install", "-y", "build-essential"], 
                        check=True, 
                        timeout=120, 
                        capture_output=True
                    )
                    _install_msg.append("✅ Build tools installed")
                except Exception as build_error:
                    _install_msg.append(f"❌ Failed to install build tools: {str(build_error)}")
                    _install_msg.append("\n**Manual fix:** Run `sudo apt-get install build-essential`")
                    _gpu_burn_path = None
                    raise Exception("Build tools installation failed")
            else:
                _install_msg.append("✅ Build tools found")
            
            # Check for CUDA toolkit in common locations
            _cuda_paths = [
                "/usr/local/cuda/bin",
                "/usr/local/cuda-12.1/bin",
                "/usr/local/cuda-12.0/bin",
                "/usr/local/cuda-11.8/bin",
                "/usr/local/cuda-11.7/bin",
                "/opt/cuda/bin",
            ]
            
            _nvcc_path = None
            _cuda_bin_dir = None
            
            # First check if nvcc is already in PATH
            _nvcc_result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
            if _nvcc_result.returncode == 0:
                _nvcc_path = _nvcc_result.stdout.strip()
                _cuda_bin_dir = os.path.dirname(_nvcc_path)
                _install_msg.append(f"✅ CUDA toolkit found in PATH: {_cuda_bin_dir}")
            else:
                # Check common CUDA locations
                for cuda_path in _cuda_paths:
                    nvcc_candidate = os.path.join(cuda_path, "nvcc")
                    if os.path.exists(nvcc_candidate):
                        _nvcc_path = nvcc_candidate
                        _cuda_bin_dir = cuda_path
                        _install_msg.append(f"✅ CUDA toolkit found: {_cuda_bin_dir}")
                        _install_msg.append(f"   (adding to PATH for this session)")
                        # Add to PATH for this process and subprocesses
                        os.environ["PATH"] = f"{_cuda_bin_dir}:{os.environ.get('PATH', '')}"
                        break
            
            if not _nvcc_path:
                _install_msg.append("⚠️ **CUDA development toolkit (nvcc) not found in:**")
                for p in _cuda_paths[:3]:  # Show first 3 paths
                    _install_msg.append(f"   - {p}")
                _install_msg.append("")
                _install_msg.append("🔧 **Attempting to install CUDA toolkit via apt-get...**")
                _install_msg.append("   ⏳ This may take 2-3 minutes (~2GB download)")
                _install_msg.append("")
                try:
                    # Check if apt is available (Debian/Ubuntu systems)
                    _apt_check = subprocess.run(["which", "apt-get"], capture_output=True, text=True)
                    if _apt_check.returncode == 0:
                        _install_msg.append("   📋 Step 1/3: Updating package lists...")
                        
                        # Update apt cache first
                        _apt_update = subprocess.run(
                            ["sudo", "apt-get", "update", "-qq"],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        
                        if _apt_update.returncode != 0:
                            _install_msg.append("   ⚠️ apt-get update had warnings (continuing anyway)")
                        else:
                            _install_msg.append("   ✅ Package lists updated")
                        
                        _install_msg.append("")
                        _install_msg.append("   📦 Step 2/3: Installing nvidia-cuda-toolkit...")
                        _install_msg.append("   (Downloading ~2GB of packages, please wait...)")
                        
                        # Install cuda toolkit
                        _cuda_install = subprocess.run(
                            ["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"],
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout
                        )
                        
                        if _cuda_install.returncode == 0:
                            _install_msg.append("   ✅ CUDA toolkit installed successfully (~5.4GB)")
                            _install_msg.append("")
                            _install_msg.append("   🔍 Step 3/3: Verifying nvcc compiler...")
                            
                            # Check if nvcc is now available
                            _nvcc_result = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
                            if _nvcc_result.returncode == 0:
                                _nvcc_path = _nvcc_result.stdout.strip()
                                _cuda_bin_dir = os.path.dirname(_nvcc_path)
                                _install_msg.append(f"   ✅ nvcc compiler found at: `{_nvcc_path}`")
                                _install_msg.append("")
                                _install_msg.append("**✨ CUDA toolkit installation complete! Proceeding with gpu-burn compilation...**")
                            else:
                                _install_msg.append("⚠️ CUDA toolkit installed but nvcc not in PATH")
                                _install_msg.append("   Using continuous PyTorch stress test instead")
                                _gpu_burn_path = None
                                raise Exception("nvcc not found after installation")
                        else:
                            _install_msg.append(f"❌ Failed to install CUDA toolkit via apt")
                            _install_msg.append(f"   Error: {_cuda_install.stderr[:200]}")
                            _install_msg.append("   Using continuous PyTorch stress test instead")
                            _gpu_burn_path = None
                            raise Exception("CUDA toolkit installation failed")
                    else:
                        _install_msg.append("⚠️ apt-get not available (non-Debian system)")
                        _install_msg.append("   Using continuous PyTorch stress test instead")
                        _gpu_burn_path = None
                        raise Exception("Cannot install CUDA toolkit automatically")
                        
                except subprocess.TimeoutExpired:
                    _install_msg.append("❌ CUDA toolkit installation timed out")
                    _install_msg.append("   Using continuous PyTorch stress test instead")
                    _gpu_burn_path = None
                    raise Exception("CUDA toolkit installation timeout")
                except Exception as install_error:
                    _install_msg.append(f"❌ Error during CUDA toolkit installation: {str(install_error)}")
                    _install_msg.append("   Using continuous PyTorch stress test instead")
                    _gpu_burn_path = None
                    raise Exception(f"CUDA toolkit installation error: {install_error}")
            
            _install_msg.append("")
            _install_msg.append("🔨 **Step 4: Compiling gpu-burn from source...**")
            
            # Compile from source
            _home_dir = os.path.expanduser("~")
            _gpu_burn_dir = os.path.join(_home_dir, "gpu-burn")
            
            if not os.path.exists(_gpu_burn_dir):
                subprocess.run(
                    ["git", "clone", "https://github.com/wilicc/gpu-burn.git", _gpu_burn_dir],
                    check=True,
                    timeout=30,
                    cwd=_home_dir,
                    capture_output=True,
                    text=True
                )
            
            # Run make with captured output for better error messages
            _make_result = subprocess.run(
                ["make"], 
                timeout=60, 
                cwd=_gpu_burn_dir,
                capture_output=True,
                text=True
            )
            
            if _make_result.returncode != 0:
                _install_msg.append(f"❌ Compilation failed (exit code {_make_result.returncode})")
                _install_msg.append(f"\n**Build output:**\n```\n{_make_result.stdout}\n{_make_result.stderr}\n```")
                _install_msg.append("\n**Common fixes:**")
                _install_msg.append("- Ensure build tools are installed: `sudo apt-get install build-essential`")
                _install_msg.append("- CUDA toolkit must be available (usually pre-installed on GPU instances)")
                _gpu_burn_path = None
            else:
                # Set path to compiled binary
                _gpu_burn_path = os.path.join(_gpu_burn_dir, "gpu_burn")
                
                if os.path.exists(_gpu_burn_path):
                    _install_msg.append(f"   ✅ Successfully compiled gpu-burn!")
                    _install_msg.append(f"   📍 Binary location: `{_gpu_burn_path}`")
                    _install_msg.append("")
                    _install_msg.append("**🎉 All steps complete! gpu-burn is ready for stress testing.**")
                else:
                    _install_msg.append(f"   ❌ Compilation succeeded but binary not found at {_gpu_burn_path}")
                    _gpu_burn_path = None
                
        except subprocess.TimeoutExpired:
            _install_msg.append(f"❌ Compilation timed out (>60s)")
            _gpu_burn_path = None
        except Exception as e:
            _install_msg.append(f"❌ Installation failed: {str(e)}")
            _gpu_burn_path = None
        
        if not _gpu_burn_path:
            test_result = mo.callout(
                mo.md("\n".join(_install_msg) + "\n\n**Falling back to PyTorch stress test** (see below)"),
                kind="warn"
            )
        else:
            test_result = mo.callout(
                mo.md("\n".join(_install_msg) + "\n\n**Ready!** Toggle the switch to start stress testing."),
                kind="success"
            )
    elif _is_running:
        # Use gpu-burn if available, otherwise fall back to PyTorch
        if _gpu_burn_path:
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
            # Continuous PyTorch fallback - runs as background process
            try:
                # Check if our background stress script is already running
                _stress_running = False
                _stress_pid = None
                for proc in psutil.process_iter(['pid', 'cmdline']):
                    try:
                        cmdline_str = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        if 'pytorch_gpu_stress.py' in cmdline_str:
                            _stress_running = True
                            _stress_pid = proc.info['pid']
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                if not _stress_running:
                    # Create continuous background stress test script
                    _stress_script = os.path.expanduser("~/pytorch_gpu_stress.py")
                    _stress_script_content = '''#!/usr/bin/env python3
import torch
import time
import sys

def stress_gpu(gpu_id):
    """Continuously stress a single GPU with maximum compute"""
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)
    
    # Try different sizes to fit in GPU memory
    for size in [16384, 12288, 8192, 4096]:
        try:
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            print(f"GPU {gpu_id}: Stress test running with {size}x{size} matrices", flush=True)
            break
        except RuntimeError:
            continue
    
    # Continuous loop - runs until killed
    iteration = 0
    while True:
        try:
            # Intensive matrix operations
            c = torch.matmul(a, b)
            d = torch.matmul(c, a)
            e = torch.matmul(d, b)
            result = torch.relu(e)
            torch.cuda.synchronize(device)
            
            # Rotate tensors to prevent optimization
            a, b = result, a
            
            iteration += 1
            if iteration % 100 == 0:
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"GPU {gpu_id} error: {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    print(f"Stressing {gpu_count} GPU(s) continuously...", flush=True)
    
    if gpu_count == 1:
        stress_gpu(0)
    else:
        import multiprocessing
        processes = []
        for gpu_id in range(gpu_count):
            p = multiprocessing.Process(target=stress_gpu, args=(gpu_id,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
'''
                    
                    with open(_stress_script, 'w') as f:
                        f.write(_stress_script_content)
                    os.chmod(_stress_script, 0o755)
                    
                    # Start background process
                    _process = subprocess.Popen(
                        ["python3", _stress_script],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True  # Detach from parent
                    )
                    _stress_pid = _process.pid
                    time.sleep(0.5)  # Give it time to start
                    
                    _status_msg = f"✅ Started continuous PyTorch stress (PID: {_stress_pid})"
                else:
                    _status_msg = f"🔥 PyTorch stress already running (PID: {_stress_pid})"
                
                _gpu_count = torch.cuda.device_count()
                test_result = mo.callout(
                    mo.md(f"""
                    🔥 **Continuous PyTorch GPU Stress Test ACTIVE!**
                    
                    **Status**: {_status_msg}  
                    **GPUs**: {_gpu_count}  
                    **Type**: Continuous background process  
                    **Intensity**: Maximum (large matrix operations)
                    
                    📊 **Runs continuously** until you toggle off  
                    📈 Watch metrics above hit 90-100% utilization  
                    🔥 Stresses ALL GPUs simultaneously
                    
                    *This runs as a background process - just like gpu-burn!*  
                    Toggle off to stop.
                    """),
                    kind="info"
                )
                
            except Exception as e:
                import traceback
                test_result = mo.callout(
                    mo.md(f"❌ **PyTorch stress test failed**: {str(e)}\n\n```\n{traceback.format_exc()}\n```"),
                    kind="danger"
                )
    else:
        # Kill any running stress processes when switch is off (using psutil from cell-2)
        _killed_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                # Kill gpu_burn, gpu-stress, or pytorch_gpu_stress processes
                if 'gpu_burn' in cmdline or 'gpu-stress' in cmdline or 'pytorch_gpu_stress.py' in cmdline:
                    proc.kill()
                    _killed_processes.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Clean up PyTorch memory if it was used
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        if _killed_processes:
            test_result = mo.callout(
                mo.md(f"🛑 **Stopped stress test** (killed PID: {', '.join(map(str, _killed_processes))})\n\nGPU will cool down now. Metrics will return to idle state."),
                kind="success"
            )
        else:
            # Show appropriate message based on gpu-burn availability
            if _gpu_burn_path:
                test_result = mo.md("""
                💡 **Toggle the switch above to start GPU stress testing**
                
                This uses the industry-standard **gpu-burn** tool:
                - Automatically stresses ALL GPUs simultaneously
                - Uses 95% of GPU memory by default
                - Double-precision floating point operations
                - Battle-tested stress testing (used in datacenters worldwide)
                - Works on L40S, A100, H100, H200, B200, and all NVIDIA GPUs
                
                **Runs in background so metrics update in real-time!**
                
                Enable auto-refresh above to watch GPUs hit 100%!
                """)
            else:
                test_result = mo.md("""
                💡 **Toggle the switch above to start GPU stress testing**
                
                **Using continuous PyTorch stress test**:
                - Runs as background process (just like gpu-burn!)
                - Maximum compute intensity with large matrix operations
                - Stresses ALL GPUs simultaneously
                - Continuous operation until you toggle off
                - 90-100% GPU utilization
                - Works without CUDA development toolkit
                
                *This is a true continuous stress test, not intermittent.*
                *Watch metrics hit 90-100% utilization on all GPUs!*
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

