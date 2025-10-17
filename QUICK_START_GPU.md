# Quick Start Guide - NVIDIA GPU Notebooks

## 🚀 Ready to Use on Any GPU Config

All notebooks work on: **L40S, A100, H100, H200, B200, RTX PRO 6000** in **1x, 2x, 4x, 8x** configurations.

## Notebooks at a Glance

| # | Notebook | Purpose | Memory | Multi-GPU |
|---|----------|---------|---------|-----------|
| 1 | `llm_finetuning_dashboard.py` | LoRA fine-tuning with GPT-2 | 8GB+ | Single GPU |
| 2 | `rapids_cudf_benchmark.py` | Pandas vs cuDF speedup | 4GB+ | Single GPU |
| 3 | `tensorrt_optimization.py` | TensorRT model optimization | 4GB+ | Single GPU |
| 4 | `stable_diffusion_trt.py` | Text-to-image generation | 8GB+ | Single GPU* |
| 5 | `nerf_training_viewer.py` | Neural radiance fields | 2GB+ | Single GPU |
| 6 | `triton_inference_server.py` | Model serving simulation | 4GB+ | Single GPU |
| 7 | `physics_informed_nn.py` | PDE solving with PINNs | 2GB+ | Single GPU |
| 8 | `graph_analytics_cugraph.py` | Graph analysis speedup | 4GB+ | Single GPU |
| 9 | `protein_structure_alphafold.py` | Protein folding demo | 2GB+ | Single GPU |
| 10 | `multi_gpu_training.py` | Distributed training | 4GB+ per | **Multi-GPU** |

*Auto-adjusts settings based on available memory

## Quick Launch

```bash
# Run any notebook
marimo run <notebook_name>.py

# Examples
marimo run llm_finetuning_dashboard.py
marimo run multi_gpu_training.py
```

## GPU Selection

### Single-GPU Notebooks (1-9)
- Always use GPU 0
- Work in any multi-GPU system
- No special configuration needed

### Multi-GPU Notebook (10)
- Automatically detects all available GPUs
- Uses PyTorch DataParallel
- Batch size auto-adjusts to GPU count

## Memory Guide

| GPU Model | Memory | Recommended Notebooks |
|-----------|--------|----------------------|
| **L40S** | 48GB | All notebooks, large batch sizes |
| **A100 40GB** | 40GB | All notebooks, medium-large batches |
| **A100 80GB** | 80GB | All notebooks, maximum settings |
| **H100** | 80GB | All notebooks, maximum performance |
| **H200** | 141GB | All notebooks, extreme batch sizes |
| **B200** | 180GB | All notebooks, future-proof |
| **RTX PRO 6000** | 48GB | All notebooks, professional use |

## Special Features

### 🎨 stable_diffusion_trt.py
**Memory-Aware Settings**:
- Detects your GPU memory on startup
- Adjusts image size options automatically
- Prevents out-of-memory errors

**What you'll see**:
- 8-12GB GPU: 512x512 images only
- 12-24GB GPU: Up to 512x768 images
- 24GB+ GPU: All sizes including 768x768

### 🚀 multi_gpu_training.py
**Adaptive Multi-GPU**:
- Batch size slider steps match GPU count
- Compares single vs multi-GPU performance
- Shows scaling efficiency charts

**Expected Speedups**:
- 2x GPU: ~1.8x faster (90% efficient)
- 4x GPU: ~3.5x faster (87% efficient)
- 8x GPU: ~6.5x faster (81% efficient)

## Troubleshooting

### Check GPU Availability
```python
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs available')"
```

### Check GPU Memory
```python
python -c "import torch; print(f'GPU 0: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

### Monitor GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or simplified
nvidia-smi
```

### Common Issues

**"CUDA out of memory"**
- Reduce batch size in notebook UI
- Close other GPU applications
- Use smaller model size option

**"No GPU detected"**
- Check: `nvidia-smi` works
- Verify: PyTorch CUDA available
- Try: Restart notebook

**"Import Error: cudf/cugraph"**
- These are optional libraries
- Notebooks fall back to CPU (pandas/NetworkX)
- Install with: `conda install -c rapidsai cudf cugraph`

## Performance Tips

### Maximize Speed
1. ✅ Use FP16/BF16 precision (enabled by default in most notebooks)
2. ✅ Increase batch size to fill GPU memory
3. ✅ Enable cuDNN benchmark (enabled automatically)
4. ✅ Use TensorRT optimization where available

### Maximize Quality
1. ✅ Increase training epochs/iterations
2. ✅ Use larger models (adjust in UI)
3. ✅ Reduce learning rate for stability
4. ✅ Monitor loss curves in real-time

## Next Steps

### For Beginners
Start with these lightweight notebooks:
1. `nerf_training_viewer.py` - Fast, visual results
2. `physics_informed_nn.py` - Educational, small models
3. `rapids_cudf_benchmark.py` - Clear speedup demonstration

### For ML Engineers
Production-focused notebooks:
1. `tensorrt_optimization.py` - Model deployment
2. `triton_inference_server.py` - Serving architecture
3. `multi_gpu_training.py` - Scaling training

### For Researchers
Advanced techniques:
1. `llm_finetuning_dashboard.py` - Parameter-efficient fine-tuning
2. `stable_diffusion_trt.py` - Generative models
3. `protein_structure_alphafold.py` - Scientific computing

## Need Help?

- 📖 See `GPU_COMPATIBILITY_MATRIX.md` for detailed specs
- 📋 See `REVIEW_SUMMARY.md` for testing details
- 🐛 See `COMPATIBILITY_REVIEW.md` for technical analysis

## Configuration Summary

✅ **Verified Compatible**:
- All 10 notebooks
- All GPU models (L40S, A100, H100, H200, B200, RTX PRO 6000)
- All configurations (1x, 2x, 4x, 8x)
- CUDA 11.4+ and 12.x

🎉 **Ready for production deployment on Brev platform!**

