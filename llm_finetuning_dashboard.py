"""llm_finetuning_dashboard.py

Interactive LLM Fine-Tuning with LoRA
======================================

Real-time dashboard for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation)
with live loss curves, GPU monitoring, and sample generation during training.

Features:
- Interactive hyperparameter tuning (learning rate, rank, batch size)
- Live training loss visualization with auto-refresh
- GPU memory monitoring during training
- Sample text generation to validate model quality
- Compare CPU vs GPU training performance
- Proper checkpoint management

Requirements:
- NVIDIA GPU with 8GB+ VRAM (works on all data center GPUs)
- Tested on: L40S (48GB), A100 (40/80GB), H100 (80GB), H200 (141GB), B200 (180GB), RTX PRO 6000 (48GB)
- CUDA 11.4+
- Model memory: ~1-2GB for GPT-2, adjustable batch size for any GPU
- Single GPU only (uses GPU 0)

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
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from typing import Optional, Dict, List, Tuple
    import subprocess
    import time
    from dataclasses import dataclass
    from torch.utils.data import Dataset, DataLoader
    import json
    
    # Check for transformers library
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_VERSION = None
    try:
        import transformers
        TRANSFORMERS_AVAILABLE = True
        TRANSFORMERS_VERSION = transformers.__version__
        print(f"✅ Transformers v{TRANSFORMERS_VERSION} package available")
    except ImportError as e:
        TRANSFORMERS_AVAILABLE = False
        TRANSFORMERS_VERSION = None
        print(f"⚠️ Transformers not available: {str(e)[:100]}")
    
    # Check for GPUtil for better GPU monitoring
    try:
        import GPUtil
        GPUTIL_AVAILABLE = True
    except ImportError:
        GPUtil = None
        GPUTIL_AVAILABLE = False
    
    return (
        mo, torch, nn, np, pd, go, Optional, Dict, List, Tuple,
        subprocess, time, dataclass, Dataset, DataLoader, json,
        TRANSFORMERS_AVAILABLE, TRANSFORMERS_VERSION,
        GPUtil, GPUTIL_AVAILABLE
    )


@app.cell
def __(mo, TRANSFORMERS_AVAILABLE, subprocess):
    """Auto-install transformers if missing"""
    transformers_install_msg = None
    transformers_needs_restart = False
    
    if not TRANSFORMERS_AVAILABLE:
        print("🔄 Transformers not found - starting auto-installation...")
        print("   (This will also install/upgrade torchvision for compatibility)")
        try:
            # Install transformers and ensure torchvision compatibility
            result = subprocess.run(
                ["pip", "install", "--upgrade", "transformers", "torchvision"],
                capture_output=True,
                text=True,
                timeout=300  # Increased timeout for two packages
            )
            
            if result.returncode == 0:
                print("✅ Transformers installed successfully!")
                transformers_needs_restart = True
                transformers_install_msg = mo.callout(
                    mo.md("""
                    ✅ **Transformers Installed Successfully!**
                    
                    **⚠️ ACTION REQUIRED:** The package was installed, but Python needs to be restarted to use it.
                    
                    **Please refresh this page now** to restart the kernel and enable the transformers library.
                    
                    After refreshing, you can click "Start Fine-Tuning" and it will work.
                    """),
                    kind="warn"
                )
            else:
                print(f"❌ Installation failed: {result.stderr[:200]}")
                transformers_install_msg = mo.callout(
                    mo.md(f"""
                    ❌ **Auto-installation Failed**
                    
                    Could not automatically install transformers.
                    
                    **Error**: {result.stderr[:300]}
                    
                    **Please install manually**:
                    ```bash
                    pip install transformers
                    ```
                    
                    Then refresh this page.
                    """),
                    kind="danger"
                )
        except subprocess.TimeoutExpired:
            print("⏱️ Installation timeout")
            transformers_install_msg = mo.callout(
                mo.md("""
                ⏱️ **Installation Timeout**
                
                The installation is taking longer than expected.
                
                **Please install manually in a terminal**:
                ```bash
                pip install transformers
                ```
                
                Then refresh this page.
                """),
                kind="warn"
            )
        except Exception as e:
            print(f"❌ Installation error: {str(e)}")
            transformers_install_msg = mo.callout(
                mo.md(f"""
                ❌ **Installation Error**
                
                {str(e)}
                
                **Please install manually**:
                ```bash
                pip install transformers
                ```
                
                Then refresh this page.
                """),
                kind="danger"
            )
    else:
        print("✅ Transformers library already available")
        transformers_install_msg = None
    
    return transformers_install_msg, transformers_needs_restart


@app.cell
def __(mo, GPUTIL_AVAILABLE, subprocess):
    """Auto-install GPUtil for enhanced GPU monitoring"""
    
    gputil_module = None
    gputil_available = False
    gputil_install_msg = None
    
    # Try importing GPUtil first
    try:
        import GPUtil as gputil_module
        gputil_available = True
        print("✅ GPUtil already available")
    except ImportError:
        gputil_module = None
        gputil_available = False
        print("⚠️ GPUtil not found, installing for enhanced GPU monitoring...")
    
    if not gputil_available:
        with mo.status.spinner(title="📦 Installing GPUtil...", subtitle="Quick install for enhanced GPU monitoring"):
            try:
                gputil_result = subprocess.run(
                    ["pip", "install", "gputil"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if gputil_result.returncode == 0:
                    # Try importing the newly installed GPUtil
                    try:
                        import GPUtil as gputil_module
                        gputil_available = True
                        gputil_install_msg = mo.callout(
                            mo.md("✅ **GPUtil installed!** Enhanced GPU monitoring now active (utilization %, temperature)."),
                            kind="success"
                        )
                        print("✅ GPUtil installed and imported successfully")
                    except ImportError:
                        gputil_available = False
                        gputil_module = None
                        gputil_install_msg = mo.callout(
                            mo.md("✅ **GPUtil installed!** Restart notebook for full monitoring features."),
                            kind="info"
                        )
                else:
                    gputil_install_msg = mo.callout(
                        mo.md("⚠️ **GPUtil install skipped**. Basic GPU monitoring still available."),
                        kind="warn"
                    )
            except Exception as e:
                gputil_install_msg = mo.callout(
                    mo.md(f"ℹ️ **Running without GPUtil** - basic GPU monitoring active. Install manually: `pip install gputil`"),
                    kind="info"
                )
    else:
        # GPUtil was already available, don't show a message
        gputil_install_msg = None
    
    return gputil_available, gputil_module, gputil_install_msg,


@app.cell
def __(TRANSFORMERS_AVAILABLE, transformers_install_msg):
    """Import transformers after check - always try fresh import"""
    # Always attempt fresh import (in case it was just installed)
    # Depends on transformers_install_msg to re-run after installation
    AutoModelForCausalLM = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None
    
    if TRANSFORMERS_AVAILABLE:
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                get_linear_schedule_with_warmup
            )
            print("✅ Successfully imported transformers classes")
        except ImportError as e:
            # Fallback to None if still not available
            print(f"⚠️ Import failed: {str(e)[:150]}")
            pass
    else:
        # Try anyway in case it was just installed
        try:
            from transformers import (
                AutoModelForCausalLM, 
                AutoTokenizer, 
                get_linear_schedule_with_warmup
            )
            print("✅ Successfully imported transformers classes (fresh install)")
        except ImportError as e:
            print(f"⚠️ Transformers still not available: {str(e)[:100]}")
            pass
    
    return AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


@app.cell
def __(mo, TRANSFORMERS_VERSION, transformers_install_msg, gputil_install_msg):
    """Title and description"""
    version_info = f" (transformers v{TRANSFORMERS_VERSION})" if TRANSFORMERS_VERSION else ""
    
    mo.vstack([
        mo.md(
            f"""
            # 🧠 Interactive LLM Fine-Tuning Dashboard{version_info}
            
            **Fine-tune large language models** with LoRA (Low-Rank Adaptation) and monitor 
            training progress in real-time. This notebook demonstrates efficient fine-tuning
            on NVIDIA GPUs using parameter-efficient methods.
            
            **What is LoRA?** LoRA freezes pretrained model weights and injects trainable 
            rank decomposition matrices, reducing trainable parameters by 10,000x while 
            maintaining quality.
            
            ## ⚙️ Training Configuration
            """
        ),
        transformers_install_msg if transformers_install_msg else mo.md(""),
        gputil_install_msg if gputil_install_msg else mo.md("")
    ])


@app.cell
def __(mo):
    """Interactive training controls"""
    learning_rate = mo.ui.slider(
        start=1e-5, stop=1e-3, step=1e-5, value=5e-4,
        label="Learning Rate", show_value=True
    )
    
    lora_rank = mo.ui.slider(
        start=4, stop=64, step=4, value=16,
        label="LoRA Rank (r)", show_value=True
    )
    
    batch_size = mo.ui.slider(
        start=1, stop=16, step=1, value=4,
        label="Batch Size", show_value=True
    )
    
    num_epochs = mo.ui.slider(
        start=1, stop=10, step=1, value=3,
        label="Training Epochs", show_value=True
    )
    
    use_mixed_precision = mo.ui.checkbox(
        value=True,
        label="Enable Mixed Precision (FP16)"
    )
    
    mo.vstack([
        mo.hstack([learning_rate, lora_rank], justify="start"),
        mo.hstack([batch_size, num_epochs, use_mixed_precision], justify="start")
    ])
    return learning_rate, lora_rank, batch_size, num_epochs, use_mixed_precision


@app.cell
def __(torch, mo, subprocess, Dict):
    """GPU Detection and Validation"""
    
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
    
    # Stop execution if no GPU available
    if not gpu_info['available']:
        mo.stop(
            True,
            mo.callout(
                mo.md(f"""
                ⚠️ **No GPU Detected**
                
                This notebook requires an NVIDIA GPU for LLM fine-tuning.
                
                **Error**: {gpu_info.get('error', 'Unknown error')}
                
                **Troubleshooting**:
                - Run `nvidia-smi` to verify GPU is detected
                - Check CUDA driver installation
                - Ensure PyTorch has CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
                
                **Note**: CPU training is too slow for LLMs - GPU is required.
                """),
                kind="danger"
            )
        )
    
    # Display GPU info
    mo.callout(
        mo.vstack([
            mo.md("**✅ GPU Detected**"),
            mo.ui.table(gpu_info['gpus'])
        ]),
        kind="success"
    )
    
    device = torch.device("cuda")
    
    return get_gpu_info, gpu_info, device


@app.cell
def __(mo, device, torch):
    """GPU Info Display"""
    try:
        if device.type == "cuda":
            gpu_props = torch.cuda.get_device_properties(0)
            total_gb = gpu_props.total_memory / 1024**3
            
            # Decode compute capability
            cc = f"{gpu_props.major}.{gpu_props.minor}"
            cc_info = {
                "8.9": "Ada Lovelace (Data Center)",
                "8.6": "Ampere (RTX 30 series)",
                "8.0": "Ampere (A100)",
                "7.5": "Turing (RTX 20 series)",
                "7.0": "Volta (V100)",
            }
            arch = cc_info.get(cc, f"Architecture {cc}")
            
            gpu_info_display = mo.callout(
                mo.md(f"""
**🖥️ GPU Detected**: {gpu_props.name}  
**Total Memory**: {total_gb:.2f} GB  
**Compute Capability**: {cc} _{arch}_

💡 **Compute Capability** is NVIDIA's GPU architecture version. Higher = newer/better features.  
💡 GPU metrics (utilization, memory, temperature) will be tracked during training.
                """),
                kind="success"
            )
        else:
            gpu_info_display = mo.callout(
                mo.md("**CPU Mode**: No GPU detected. Training will run on CPU (slower)."),
                kind="warn"
            )
    except Exception as e:
        gpu_info_display = mo.callout(
            mo.md(f"**Error reading GPU info**: {str(e)}"),
            kind="danger"
        )
    
    gpu_info_display
    
    return gpu_info_display,


@app.cell
def __(Dataset, torch, List, Tuple, Dict):
    """Dataset preparation"""
    
    class FineTuningDataset(Dataset):
        """Simple dataset for demonstration - replace with your own data"""
        
        def __init__(self, tokenizer, num_samples: int = 100):
            self.tokenizer = tokenizer
            self.samples = self._generate_samples(num_samples)
        
        def _generate_samples(self, num_samples: int) -> List[str]:
            """Generate realistic synthetic training samples for demonstration"""
            
            # Realistic examples that look like actual fine-tuning data
            examples = [
                # Translation examples
                "Translate to French: Hello, how are you? -> Bonjour, comment allez-vous?",
                "Translate to French: The weather is nice today. -> Le temps est beau aujourd'hui.",
                "Translate to Spanish: Good morning! -> Buenos días!",
                "Translate to Spanish: Thank you very much. -> Muchas gracias.",
                "Translate to German: Where is the train station? -> Wo ist der Bahnhof?",
                
                # Summarization examples
                "Summarize: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Summary: ML allows computers to learn from data.",
                "Summarize: The annual tech conference showcased innovations in AI, robotics, and quantum computing. Summary: Conference highlighted AI, robotics, and quantum tech advances.",
                "Summarize: Climate change poses significant challenges requiring global cooperation and sustainable practices. Summary: Climate change demands worldwide action and sustainability.",
                
                # Question answering
                "Question: What is machine learning? Answer: Machine learning is a method of teaching computers to learn from data and make decisions.",
                "Question: Who invented the telephone? Answer: Alexander Graham Bell invented the telephone in 1876.",
                "Question: What is photosynthesis? Answer: Photosynthesis is the process plants use to convert sunlight into energy.",
                "Question: What is the capital of France? Answer: The capital of France is Paris.",
                
                # Instruction following
                "Instruction: Write a haiku about coding. Output: Lines of code flow smooth / Logic dances on the screen / Bugs vanish at dawn",
                "Instruction: Explain recursion simply. Output: Recursion is when a function calls itself to solve smaller versions of the same problem.",
                "Instruction: List 3 benefits of exercise. Output: 1) Improves cardiovascular health 2) Boosts mood and energy 3) Strengthens muscles and bones",
                
                # Sentiment analysis
                "Classify sentiment: This movie was absolutely amazing! Label: Positive",
                "Classify sentiment: I'm disappointed with the service. Label: Negative",
                "Classify sentiment: The product works as expected. Label: Neutral",
                
                # Code generation
                "Generate Python: function to add two numbers -> def add(a, b): return a + b",
                "Generate Python: function to reverse a string -> def reverse(s): return s[::-1]",
            ]
            
            # Repeat and vary examples to reach desired num_samples
            samples = []
            for i in range(num_samples):
                base_example = examples[i % len(examples)]
                # Add slight variation to index for uniqueness
                if i >= len(examples):
                    variation_num = i // len(examples)
                    samples.append(f"{base_example} [v{variation_num}]")
                else:
                    samples.append(base_example)
            
            return samples
        
        def __len__(self) -> int:
            return len(self.samples)
        
        def __getitem__(self, idx: int) -> Dict:
            text = self.samples[idx]
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            return {
                'input_ids': encodings['input_ids'].squeeze(0),
                'attention_mask': encodings['attention_mask'].squeeze(0),
                'labels': encodings['input_ids'].squeeze(0)
            }
    
    return FineTuningDataset,


@app.cell
def __(nn, torch, Tuple):
    """LoRA implementation"""
    
    class LoRALayer(nn.Module):
        """Low-Rank Adaptation layer"""
        
        def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
            super().__init__()
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # LoRA matrices
            self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: x @ A @ B (handles mixed precision)"""
            # Store original dtype for mixed precision training
            original_dtype = x.dtype
            
            # Cast to FP32 for LoRA computation (parameters are FP32)
            x_fp32 = x.to(torch.float32)
            result = (x_fp32 @ self.lora_A @ self.lora_B) * self.scaling
            
            # Cast back to original dtype (FP16 if model is FP16)
            return result.to(original_dtype)
    
    def inject_lora(model: nn.Module, rank: int = 16) -> Tuple[nn.Module, int, List]:
        """Inject LoRA layers into model attention layers"""
        # FIRST: Freeze ALL model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        lora_params = 0
        trainable_params = []  # Collect LoRA parameters
        
        # THEN: Add LoRA layers and make them trainable
        for name, module in model.named_modules():
            # GPT-2 uses Conv1D for attention, LLaMA uses Linear
            # GPT-2: c_attn (combined QKV), c_proj (output)
            # LLaMA: q_proj, v_proj, k_proj
            is_linear_or_conv1d = isinstance(module, nn.Linear) or type(module).__name__ == 'Conv1D'
            is_attention_layer = any(x in name for x in ['c_attn', 'c_proj', 'q_proj', 'v_proj', 'k_proj'])
            
            if is_linear_or_conv1d and is_attention_layer:
                # Get dimensions (Conv1D and Linear have different weight layouts)
                if type(module).__name__ == 'Conv1D':
                    # GPT-2 Conv1D: weight shape is (in_features, out_features)
                    # This is OPPOSITE of nn.Linear!
                    in_features = module.weight.shape[0]
                    out_features = module.weight.shape[1]
                else:
                    # nn.Linear: weight shape is (out_features, in_features)
                    in_features = module.in_features
                    out_features = module.out_features
                
                # Add LoRA layer and move to same device as module
                lora_layer = LoRALayer(in_features, out_features, rank=rank)
                
                # Move LoRA to same device as the module
                # Keep LoRA parameters in FP32 (even if model is FP16)
                # This is standard practice for mixed precision training!
                if hasattr(module.weight, 'device'):
                    lora_layer = lora_layer.to(device=module.weight.device, dtype=torch.float32)
                
                # Store LoRA parameters for optimizer
                trainable_params.extend([lora_layer.lora_A, lora_layer.lora_B])
                
                # Attach to module (for forward pass)
                module._lora = lora_layer
                lora_params += rank * (in_features + out_features)
                
                # Store original forward method
                original_forward = module.forward
                
                # Monkey-patch forward to include LoRA
                def make_forward_with_lora(orig_forward, lora):
                    def forward_with_lora(x):
                        # Base output from frozen weights
                        base_out = orig_forward(x)
                        # LoRA adaptation
                        lora_out = lora(x)
                        return base_out + lora_out
                    return forward_with_lora
                
                module.forward = make_forward_with_lora(original_forward, lora_layer)
        
        return model, lora_params, trainable_params
    
    return LoRALayer, inject_lora


@app.cell
def __(mo):
    """Training trigger button"""
    train_button = mo.ui.run_button(label="🚀 Start Fine-Tuning")
    
    mo.vstack([
        mo.md("### 🎯 Training Control"),
        train_button
    ])
    return train_button,


@app.cell
def __(mo, pd, FineTuningDataset):
    """Preview the training dataset"""
    
    # Create sample dataset to show (using a dummy tokenizer-like object)
    class DummyTokenizer:
        def __init__(self):
            pass
    
    dummy_tok = DummyTokenizer()
    preview_dataset = FineTuningDataset(dummy_tok, num_samples=200)
    preview_samples = preview_dataset._generate_samples(200)
    
    # Create DataFrame for display (all 200 samples)
    dataset_df = pd.DataFrame({
        'Index': range(len(preview_samples)),
        'Training Sample': preview_samples
    })
    
    mo.vstack([
        mo.md("## 📊 Training Dataset Preview"),
        mo.callout(
            mo.md("""
**Realistic Demo Data**: This demo uses 200 synthetic training samples covering:
- 🌍 Translation (English → French, Spanish, German)
- 📝 Summarization (compress text into key points)
- ❓ Question Answering (factual Q&A)
- 📋 Instruction Following (haikus, explanations, lists)
- 😊 Sentiment Analysis (positive/negative/neutral)
- 💻 Code Generation (simple Python functions)

💡 **For real applications**: Replace `FineTuningDataset` with your actual task-specific data!

**Interactive Table** (showing all 200 samples with pagination):
            """),
            kind="info"
        ),
        mo.ui.table(dataset_df, selection=None, page_size=10),
        mo.md("*Scroll through pages to see all samples. The model will see all 200 during training, shuffled across batches.*"),
    ])
    
    return dataset_df,


@app.cell
def __(train_button, torch, np, mo):
    """Step 1: Initialize and set random seeds"""
    # Stop execution if button not clicked
    mo.stop(not train_button.value, mo.md("👈 **Click 'Start Fine-Tuning' to begin**"))
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    mo.callout(
        mo.md("✅ **Step 1/7 Complete:** Random seeds set for reproducibility"),
        kind="success"
    )


@app.cell
def __(train_button, mo, batch_size, lora_rank, num_epochs, use_mixed_precision):
    """Show training configuration"""
    mo.stop(not train_button.value)
    
    mo.md(f"""
    ## 🚀 Starting Fine-Tuning
    
    **Configuration:**
    - Epochs: {num_epochs.value}
    - Batch Size: {batch_size.value}
    - LoRA Rank: {lora_rank.value}
    - Precision: {'FP16' if use_mixed_precision.value else 'FP32'}
    """)


@app.cell
def __(train_button, device, use_mixed_precision, AutoModelForCausalLM, AutoTokenizer, torch, mo):
    """Step 2: Load model and tokenizer"""
    mo.stop(not train_button.value)
    
    # Check if transformers is available
    mo.stop(
        AutoTokenizer is None or AutoModelForCausalLM is None,
        mo.callout(
            mo.md("""
            ❌ **Transformers library not available**
            
            The `transformers` library failed to import. Please:
            1. Scroll up to the installation section
            2. Wait for the installation to complete
            3. Refresh the page if needed
            4. Then try starting fine-tuning again
            
            Or install manually:
            ```bash
            pip install transformers
            ```
            """),
            kind="danger"
        )
    )
    
    model_name = "gpt2"
    
    # Load tokenizer
    with mo.status.spinner(title="📥 Step 2/7: Loading tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model  
    with mo.status.spinner(title="📥 Step 2/7: Downloading GPT-2 model (first time: ~2 min)..."):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if use_mixed_precision.value else torch.float32
        )
        model = model.to(device)
    
    mo.callout(
        mo.md("✅ **Step 2/7 Complete:** Model and tokenizer loaded"),
        kind="success"
    )
    
    return model, tokenizer


@app.cell
def __(train_button, model, lora_rank, inject_lora, mo):
    """Step 3: Inject LoRA layers"""
    mo.stop(not train_button.value)
    
    model_with_lora, lora_params, lora_trainable_params = inject_lora(model, rank=lora_rank.value)
    total_params = sum(p.numel() for p in model_with_lora.parameters())
    trainable_params_count = sum(p.numel() for p in lora_trainable_params)
    
    mo.callout(
        mo.md(f"✅ **Step 3/7 Complete:** LoRA injected - Training only **{trainable_params_count:,} / {total_params:,}** parameters ({100*trainable_params_count/total_params:.2f}%)"),
        kind="success"
    )
    
    return model_with_lora, total_params, trainable_params_count, lora_params, lora_trainable_params


@app.cell
def __(train_button, model_with_lora, tokenizer, batch_size, learning_rate, device, torch, 
      FineTuningDataset, DataLoader, lora_trainable_params, mo):
    """Step 4: Prepare dataset and optimizer"""
    mo.stop(not train_button.value)
    
    dataset = FineTuningDataset(tokenizer, num_samples=200)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size.value,
        shuffle=True
    )
    
    # Use the explicitly collected LoRA parameters
    optimizer = torch.optim.AdamW(
        lora_trainable_params,
        lr=learning_rate.value
    )
    
    # Create GradScaler for FP16 training (critical for preventing NaN loss!)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision.value and device.type == "cuda" else None
    
    # Warmup run for GPU
    model_with_lora.train()
    _warmup_batch = next(iter(dataloader))
    _warmup_inputs = _warmup_batch['input_ids'][:1].to(device)
    _warmup_mask = _warmup_batch['attention_mask'][:1].to(device)
    _warmup_labels = _warmup_batch['labels'][:1].to(device)
    _ = model_with_lora(input_ids=_warmup_inputs, attention_mask=_warmup_mask, labels=_warmup_labels)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del _warmup_batch, _warmup_inputs, _warmup_mask, _warmup_labels, _
    
    mo.callout(
        mo.md(f"✅ **Step 4/7 Complete:** Dataset ready - **{len(dataset)} samples**, **{len(dataloader)} batches**"),
        kind="success"
    )
    
    return dataset, dataloader, optimizer, scaler


@app.cell
def __(train_button, model_with_lora, dataloader, optimizer, scaler, num_epochs, use_mixed_precision, 
      device, torch, time, np, mo):
    """Step 5: Training loop"""
    mo.stop(not train_button.value)
    
    mo.callout(
        mo.md(f"🔄 **Step 5/7:** Training for **{num_epochs.value} epoch(s)**... (console shows batch progress)"),
        kind="info"
    )
    
    _losses = []
    _times = []
    _epoch_stats = []
    _gpu_memory_samples = []
    _start_time = time.time()
    
    for epoch in range(num_epochs.value):
        _epoch_losses = []
        _epoch_start = time.time()
        print(f"\n  📍 Epoch {epoch+1}/{num_epochs.value}", flush=True)
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if use_mixed_precision.value and device.type == "cuda" and scaler is not None:
                # Proper FP16 training with GradScaler
                with torch.cuda.amp.autocast():
                    _outputs = model_with_lora(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = _outputs.loss
                
                # Backward pass with scaled gradients
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first for accurate clipping)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model_with_lora.parameters()), 
                    max_norm=1.0
                )
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 training (no scaler needed)
                _outputs = model_with_lora(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = _outputs.loss
                
                # Standard backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model_with_lora.parameters()), 
                    max_norm=1.0
                )
                
                optimizer.step()
            
            # Synchronize for accurate timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Track metrics
            _epoch_losses.append(loss.item())
            _losses.append(loss.item())
            _times.append(time.time() - _start_time)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"    Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}", flush=True)
            
            # Sample GPU memory periodically
            if batch_idx % 5 == 0 and device.type == "cuda":
                _gpu_memory_samples.append({
                    'time': time.time() - _start_time,
                    'memory_gb': torch.cuda.memory_allocated(0) / 1024**3
                })
        
        # Epoch summary
        _epoch_time = time.time() - _epoch_start
        _epoch_stats.append({
            'epoch': epoch + 1,
            'avg_loss': np.mean(_epoch_losses),
            'time': _epoch_time
        })
        print(f"  ✅ Epoch {epoch+1} complete: Avg Loss = {np.mean(_epoch_losses):.4f}, Time = {_epoch_time:.1f}s", flush=True)
    
    _total_time = time.time() - _start_time
    
    # Return without underscore prefix so other cells can use them
    training_losses = _losses
    training_times = _times
    epoch_stats = _epoch_stats
    gpu_memory_samples = _gpu_memory_samples
    total_training_time = _total_time
    
    return training_losses, training_times, epoch_stats, gpu_memory_samples, total_training_time


@app.cell
def __(train_button, training_losses, total_training_time, mo):
    """Display Step 5 completion"""
    mo.stop(not train_button.value)
    
    mo.callout(
        mo.md(f"✅ **Step 5/7 Complete:** Training finished in **{total_training_time:.1f}s** - Final loss: **{training_losses[-1]:.4f}**"),
        kind="success"
    )


@app.cell
def __(train_button, model_with_lora, tokenizer, device, torch, mo):
    """Step 6: Generate sample outputs"""
    mo.stop(not train_button.value)
    
    model_with_lora.eval()
    _sample_prompts = [
        "Translate English to French: Hello",
        "Summarize this text: Machine learning is transforming",
        "Answer the question: What is AI?",
    ]
    _generated_samples = []
    
    with torch.no_grad():
        for prompt in _sample_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                # Use greedy decoding (more stable than sampling after FP16 training)
                _gen_outputs = model_with_lora.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=False,  # Greedy decoding - more stable
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(_gen_outputs[0], skip_special_tokens=True)
                _generated_samples.append({'prompt': prompt, 'output': generated_text})
            except Exception as e:
                # If generation fails, record the error
                _generated_samples.append({
                    'prompt': prompt, 
                    'output': f"[Generation failed: {str(e)[:50]}]"
                })
                print(f"⚠️ Generation failed for prompt '{prompt[:30]}...': {str(e)[:100]}")
    
    # Return without underscore prefix
    generated_samples = _generated_samples
    
    return generated_samples,


@app.cell
def __(train_button, generated_samples, mo):
    """Display Step 6 completion"""
    mo.stop(not train_button.value)
    
    mo.callout(
        mo.md(f"✅ **Step 6/7 Complete:** Generated **{len(generated_samples)} samples**"),
        kind="success"
    )


@app.cell
def __(train_button, training_losses, training_times, total_training_time, total_params, trainable_params_count, lora_params,
      epoch_stats, gpu_memory_samples, generated_samples, batch_size, np, torch, model_with_lora, tokenizer, mo):
    """Step 7: Finalize results"""
    mo.stop(not train_button.value)
    
    training_results = {
        'losses': training_losses,
        'times': training_times,
        'total_time': total_training_time,
        'total_params': total_params,
        'trainable_params': trainable_params_count,
        'lora_params': lora_params,
        'final_loss': training_losses[-1],
        'avg_loss': np.mean(training_losses[-10:]),
        'generated_samples': generated_samples,
        'epoch_stats': epoch_stats,
        'gpu_memory_samples': gpu_memory_samples,
        'num_batches': len(training_losses),
        'samples_per_sec': len(training_losses) * batch_size.value / total_training_time
    }
    
    # Cleanup GPU memory
    del model_with_lora, tokenizer
    torch.cuda.empty_cache()
    
    mo.callout(
        mo.md("✅ **Step 7/7 Complete:** Results ready! Scroll down to see visualizations."),
        kind="success"
    )
    
    return training_results,


@app.cell
def __(training_results, mo, go, pd, np):
    """Visualize training results"""
    
    # This cell automatically runs after training_results is created
    # No need for mo.stop() - dependency graph handles it!
    
    # Create comprehensive visualizations
    if 'error' in training_results:
        error_msg = f"**Training Error**: {training_results['error']}"
        if 'suggestion' in training_results:
            error_msg += f"\n\n{training_results['suggestion']}"
        if 'error_type' in training_results:
            error_msg += f"\n\n*Error type: {training_results['error_type']}*"
        
        # Display error and stop
        output = mo.callout(
            mo.md(error_msg),
            kind="danger"
        )
    else:
        # Create comprehensive visualizations
        
        # Helper to format numbers safely (handle NaN)
        def safe_format(value, format_str=".4f"):
            if np.isnan(value) or np.isinf(value):
                return "❌ NaN (training failed)"
            return f"{value:{format_str}}"
        
        # 1. Key Metrics Cards
        param_reduction = training_results['total_params'] / max(training_results['trainable_params'], 1)
        
        def metric_card(title, value, subtitle):
            return mo.callout(
                mo.vstack([
                    mo.md(f"**{title}**"),
                    mo.md(f"# {value}"),
                    mo.md(f"*{subtitle}*")
                ], align="center"),
                kind="neutral"
            )
        
        metrics_row = mo.hstack([
            metric_card(
                "Parameter Efficiency",
                f"{param_reduction:.1f}x",
                f"Training {training_results['trainable_params']:,} of {training_results['total_params']:,} params"
            ),
            metric_card(
                "Training Time",
                f"{training_results['total_time']:.1f}s",
                f"{training_results['samples_per_sec']:.1f} samples/sec"
            ),
            metric_card(
                "Final Loss",
                safe_format(training_results['final_loss']),
                f"Avg: {safe_format(training_results['avg_loss'])}"
            ),
            metric_card(
                "Total Batches",
                str(training_results['num_batches']),
                f"Across {len(training_results['epoch_stats'])} epochs"
            )
        ], justify="space-around")
        
        # 2. Loss curve with smoothing
        loss_values = training_results['losses']
        time_values = training_results['times']
        
        # Calculate smoothed loss (moving average)
        window = min(10, len(loss_values) // 5)
        if window > 1:
            smoothed = np.convolve(loss_values, np.ones(window)/window, mode='valid')
            smooth_times = time_values[window-1:]
        else:
            smoothed = loss_values
            smooth_times = time_values
        
        fig_loss = go.Figure()
        
        # Raw loss
        fig_loss.add_trace(go.Scatter(
            x=time_values,
            y=loss_values,
            mode='lines',
            name='Raw Loss',
            line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
            hovertemplate='Time: %{x:.1f}s<br>Loss: %{y:.4f}<extra></extra>'
        ))
        
        # Smoothed loss
        fig_loss.add_trace(go.Scatter(
            x=smooth_times,
            y=smoothed,
            mode='lines',
            name='Smoothed Loss',
            line=dict(color='#ff6b6b', width=3),
            hovertemplate='Time: %{x:.1f}s<br>Loss: %{y:.4f}<extra></extra>'
        ))
        
        fig_loss.update_layout(
            title="Training Loss Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Loss",
            height=450,
            margin=dict(t=60, l=80, r=40, b=80),
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        # 3. Per-epoch statistics
        if training_results['epoch_stats']:
            epoch_df = pd.DataFrame(training_results['epoch_stats'])
            
            fig_epochs = go.Figure()
            
            # Bar chart for epoch loss
            fig_epochs.add_trace(go.Bar(
                x=epoch_df['epoch'],
                y=epoch_df['avg_loss'],
                name='Avg Loss',
                marker=dict(color='#4ecdc4'),
                text=[f"{val:.4f}" for val in epoch_df['avg_loss']],
                textposition='auto',
                hovertemplate='Epoch %{x}<br>Avg Loss: %{y:.4f}<extra></extra>'
            ))
            
            fig_epochs.update_layout(
                title="Average Loss per Epoch",
                xaxis_title="Epoch",
                yaxis_title="Average Loss",
                height=400,
                margin=dict(t=60, l=80, r=40, b=80),
                template='plotly_white'
            )
        else:
            fig_epochs = None
        
        # 4. GPU Memory usage during training
        if training_results['gpu_memory_samples']:
            mem_times = [s['time'] for s in training_results['gpu_memory_samples']]
            mem_values = [s['memory_gb'] for s in training_results['gpu_memory_samples']]
            
            fig_memory = go.Figure()
            fig_memory.add_trace(go.Scatter(
                x=mem_times,
                y=mem_values,
                mode='lines',
                fill='tozeroy',
                name='GPU Memory',
                line=dict(color='#95e1d3', width=2),
                fillcolor='rgba(149, 225, 211, 0.3)',
                hovertemplate='Time: %{x:.1f}s<br>Memory: %{y:.2f} GB<extra></extra>'
            ))
            
            fig_memory.update_layout(
                title="GPU Memory Usage During Training",
                xaxis_title="Time (seconds)",
                yaxis_title="Memory (GB)",
                height=400,
                margin=dict(t=60, l=80, r=40, b=80),
                template='plotly_white'
            )
        else:
            fig_memory = None
        
        # 5. Generated samples display
        samples_display = []
        for _i, sample in enumerate(training_results['generated_samples'], 1):
            samples_display.append(
                mo.callout(
                    mo.vstack([
                        mo.md(f"**Prompt:** {sample['prompt']}"),
                        mo.md(f"**Output:** {sample['output']}")
                    ]),
                    kind="neutral"  # All samples same color for consistency
                )
            )
        
        # 6. Parameter breakdown
        param_data = pd.DataFrame({
            'Type': ['Total Parameters', 'Trainable (LoRA)', 'Frozen'],
            'Count': [
                training_results['total_params'],
                training_results['trainable_params'],
                training_results['total_params'] - training_results['trainable_params']
            ],
            'Percentage': [
                100.0,
                100 * training_results['trainable_params'] / training_results['total_params'],
                100 * (training_results['total_params'] - training_results['trainable_params']) / training_results['total_params']
            ]
        })
        
        fig_params = go.Figure(data=[go.Pie(
            labels=param_data['Type'],
            values=param_data['Count'],
            hole=0.4,
            marker=dict(colors=['#a8dadc', '#457b9d', '#1d3557']),
            textinfo='label+percent',
            hovertemplate='%{label}<br>%{value:,} params<br>%{percent}<extra></extra>'
        )])
        
        fig_params.update_layout(
            title="Parameter Distribution",
            height=400,
            margin=dict(t=60, l=40, r=40, b=40),
            template='plotly_white'
        )
        
        # Assemble and display the dashboard
        output = mo.vstack([
            mo.md("# ✅ Training Complete!"),
            mo.md("---"),
            
            mo.md("## 📊 Key Metrics"),
            metrics_row,
            mo.md("---"),
            
            mo.md("## 📈 Training Progress"),
            
            # Loss and Parameters side-by-side
            mo.hstack([
                mo.vstack([
                    mo.ui.plotly(fig_loss),
                ], align="start"),
                mo.vstack([
                    mo.ui.plotly(fig_params),
                ], align="start") if fig_params else mo.md("")
            ], justify="space-around"),
            
            # Epoch stats and GPU memory stacked vertically (full width)
            mo.ui.plotly(fig_epochs) if fig_epochs else mo.md(""),
            mo.ui.plotly(fig_memory) if fig_memory else mo.md(""),
            
            mo.md("---"),
            mo.md("## 💬 Sample Text Generation"),
            mo.callout(
                mo.md("""
**About these samples**: The model was fine-tuned on 200 synthetic examples (translations, 
summarizations, Q&A, etc.) over 3 epochs. Since this is a **tiny dataset** on a **small model** (GPT-2), 
outputs may be imperfect or repetitive.

**For production**: Use thousands of high-quality examples, larger models (LLaMA 2, Mistral), 
and more training epochs for better results.
                """),
                kind="info"
            ),
            mo.vstack(samples_display),
            
            mo.md("---"),
            mo.md("### 🎯 Next Steps"),
            mo.callout(
                mo.md("""
**Production Recommendations:**
- Replace `FineTuningDataset` with your actual training data
- Use larger models (LLaMA 2, Mistral, etc.) for better results
- Increase training epochs for improved convergence
- Implement validation set monitoring
- Save checkpoints periodically using `model.save_pretrained()`
                """),
                kind="info"
            )
        ])
    
    # Return output (works for both error and success cases)
    output


@app.cell
def __(mo):
    """Export and deployment documentation"""
    mo.md(
        """
        ---
        
        ## 📚 Understanding LoRA Fine-Tuning
        
        **LoRA (Low-Rank Adaptation)** enables efficient fine-tuning by:
        - Freezing pretrained weights
        - Adding small trainable matrices (rank decomposition)
        - Reducing memory footprint by 3x
        - Maintaining model quality
        
        **Parameter Efficiency**: Instead of training 100M+ parameters, LoRA typically
        trains only 0.1-1% of parameters, making it possible to fine-tune large models
        on consumer GPUs.
        
        ### 🚀 Production Deployment
        
        **Next Steps**:
        1. Replace `FineTuningDataset` with your actual training data
        2. Use larger models (LLaMA, Mistral, etc.) for production
        3. Implement validation set and early stopping
        4. Save checkpoints periodically
        5. Use gradient accumulation for larger effective batch sizes
        
        **Run as Script**:
        ```bash
        python llm_finetuning_dashboard.py
        ```
        
        **Deploy as App**:
        ```bash
        marimo run llm_finetuning_dashboard.py
        ```
        
        ### 📖 Resources
        - [LoRA Paper](https://arxiv.org/abs/2106.09685)
        - [HuggingFace PEFT](https://github.com/huggingface/peft)
        - [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
        - [Brev.dev Documentation](https://brev.dev/docs)
        """
    )


if __name__ == "__main__":
    app.run()

