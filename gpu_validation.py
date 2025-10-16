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
    else:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                gpu_status = mo.callout(
                    mo.md("❌ **No NVIDIA GPUs detected**"),
                    kind="danger"
                )
            else:
                gpus = GPUtil.getGPUs()
                gpu_info_lines = [f"✅ **{device_count} NVIDIA GPU(s) detected**\n"]
                
                for i, gpu in enumerate(gpus):
                    gpu_info_lines.append(f"\n### GPU {i}: {gpu.name}")
                    gpu_info_lines.append(f"- **Memory**: {gpu.memoryTotal} MB")
                    gpu_info_lines.append(f"- **Driver Version**: {gpu.driver}")
                    gpu_info_lines.append(f"- **UUID**: {gpu.uuid}")
                
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
    
    gpu_status
    return device_count, gpu_info_lines, gpu_status, gpus, i


@app.cell
def __(mo):
    mo.md("""## 📊 Real-time GPU Metrics""")
    return


@app.cell
def __(GPUtil, mo):
    # Refresh button
    refresh_button = mo.ui.button(label="🔄 Refresh Metrics", kind="success")
    refresh_button
    return refresh_button,


@app.cell
def __(GPUtil, alt, mo, pd, refresh_button):
    # Trigger on button click
    refresh_button.value
    
    try:
        gpus = GPUtil.getGPUs()
        
        if not gpus:
            gpu_metrics = mo.callout(
                mo.md("No GPUs available to monitor"),
                kind="warn"
            )
        else:
            # Create metrics for each GPU
            metrics_data = []
            for gpu in gpus:
                metrics_data.append({
                    'GPU': f"GPU {gpu.id}",
                    'Metric': 'Utilization',
                    'Value': gpu.load * 100,
                    'Unit': '%'
                })
                metrics_data.append({
                    'GPU': f"GPU {gpu.id}",
                    'Metric': 'Memory Used',
                    'Value': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'Unit': '%'
                })
                metrics_data.append({
                    'GPU': f"GPU {gpu.id}",
                    'Metric': 'Temperature',
                    'Value': gpu.temperature,
                    'Unit': '°C'
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create bar chart
            chart = alt.Chart(df_metrics).mark_bar().encode(
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
            for gpu in gpus:
                card_content = mo.md(f"""
                ### GPU {gpu.id}: {gpu.name}
                
                **Utilization**: {gpu.load * 100:.1f}%  
                **Memory**: {gpu.memoryUsed:.0f} MB / {gpu.memoryTotal:.0f} MB ({(gpu.memoryUsed/gpu.memoryTotal)*100:.1f}%)  
                **Temperature**: {gpu.temperature}°C  
                **Power Draw**: {getattr(gpu, 'powerDraw', 'N/A')} W / {getattr(gpu, 'powerLimit', 'N/A')} W
                """)
                gpu_cards.append(card_content)
            
            gpu_metrics = mo.vstack([
                mo.ui.altair_chart(chart),
                mo.md("### Detailed Metrics"),
                mo.hstack(gpu_cards)
            ])
    except Exception as e:
        gpu_metrics = mo.callout(
            mo.md(f"Error reading GPU metrics: {str(e)}"),
            kind="danger"
        )
    
    gpu_metrics
    return card_content, chart, df_metrics, gpu, gpu_cards, gpu_metrics, gpus, metrics_data


@app.cell
def __(mo):
    mo.md("""## 🧪 GPU Compute Test""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        Run a simple matrix multiplication test to verify GPU compute capability.
        """
    )
    return


@app.cell
def __(mo):
    # Test button
    test_button = mo.ui.button(label="▶️ Run GPU Test", kind="success")
    test_button
    return test_button,


@app.cell
def __(mo, test_button, time, torch, torch_available):
    if test_button.value:
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
        else:
            try:
                # Matrix size for test
                size = 10000
                
                # CPU test
                cpu_tensor_a = torch.randn(size, size)
                cpu_tensor_b = torch.randn(size, size)
                
                cpu_start = time.time()
                cpu_result = torch.matmul(cpu_tensor_a, cpu_tensor_b)
                cpu_time = time.time() - cpu_start
                
                # GPU test
                device = torch.device("cuda:0")
                gpu_tensor_a = torch.randn(size, size, device=device)
                gpu_tensor_b = torch.randn(size, size, device=device)
                
                # Warm up
                _ = torch.matmul(gpu_tensor_a, gpu_tensor_b)
                torch.cuda.synchronize()
                
                # Actual test
                gpu_start = time.time()
                gpu_result = torch.matmul(gpu_tensor_a, gpu_tensor_b)
                torch.cuda.synchronize()
                gpu_time = time.time() - gpu_start
                
                speedup = cpu_time / gpu_time
                
                test_result = mo.callout(
                    mo.md(f"""
                    ✅ **GPU Compute Test Successful!**
                    
                    Matrix Multiplication ({size}x{size})
                    - **CPU Time**: {cpu_time:.4f} seconds
                    - **GPU Time**: {gpu_time:.4f} seconds
                    - **Speedup**: {speedup:.2f}x faster on GPU
                    
                    Your GPU is working correctly! 🎉
                    """),
                    kind="success"
                )
            except Exception as e:
                test_result = mo.callout(
                    mo.md(f"❌ **GPU test failed**: {str(e)}"),
                    kind="danger"
                )
    else:
        test_result = mo.md("Click the button above to run the GPU compute test.")
    
    test_result
    return (
        cpu_result,
        cpu_start,
        cpu_tensor_a,
        cpu_tensor_b,
        cpu_time,
        device,
        gpu_result,
        gpu_start,
        gpu_tensor_a,
        gpu_tensor_b,
        gpu_time,
        size,
        speedup,
        test_result,
    )


@app.cell
def __(mo):
    mo.md("""## 🛠️ nvidia-smi Output""")
    return


@app.cell
def __(mo, subprocess):
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], text=True)
        nvidia_smi_display = mo.md(f"""
        ```
        {nvidia_smi_output}
        ```
        """)
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

