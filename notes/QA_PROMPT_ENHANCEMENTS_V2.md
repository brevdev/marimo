# QA Prompt Enhancements V2.0

**Based on**: LLM Fine-Tuning Dashboard Development (October 2025)  
**Key Learnings**: FP16 training, LoRA implementation, Marimo patterns, Educational documentation

---

## New Sections to Add to QA Checklist

### 11. MIXED PRECISION TRAINING (FP16/BF16)

#### GradScaler Requirements
- [ ] **GradScaler created** when using FP16 training (`torch.cuda.amp.GradScaler()`)
- [ ] **Trainable parameters are FP32**, not FP16 (optimizer requirement)
- [ ] **Model weights can be FP16** (memory efficiency)
- [ ] **Forward pass uses autocast()** for FP16 computation
- [ ] **Backward pass uses scaler.scale(loss).backward()**
- [ ] **Gradient unscaling** before clipping: `scaler.unscale_(optimizer)`
- [ ] **Gradient clipping** after unscaling
- [ ] **Optimizer step** uses scaler: `scaler.step(optimizer)`
- [ ] **Scaler update** after step: `scaler.update()`

```python
# CORRECT FP16 Training Pattern ✅
scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

if use_fp16:
    with torch.cuda.amp.autocast():
        loss = model(...)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
else:
    loss = model(...)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
```

#### Dtype Management
- [ ] **Check that model and data dtypes match** during forward pass
- [ ] **Handle dtype conversion** in custom layers if mixing FP16 model with FP32 parameters
- [ ] **Test for NaN loss** - primary symptom of FP16 issues
- [ ] **Gradient clipping** is essential with FP16 (prevents NaN)

#### Common FP16 Errors
- [ ] `ValueError: Attempting to unscale FP16 gradients` → trainable params must be FP32
- [ ] `RuntimeError: expected same dtype` → model FP16 but params FP32, need conversion layer
- [ ] NaN loss → missing GradScaler or gradient clipping

---

### 12. LORA IMPLEMENTATION

#### Architecture-Specific Layer Detection
- [ ] **GPT-2 uses Conv1D** with shape `(in_features, out_features)` - OPPOSITE of nn.Linear!
- [ ] **LLaMA/Mistral use nn.Linear** with shape `(out_features, in_features)`
- [ ] **Check layer type** before accessing dimensions
- [ ] **Handle both Conv1D and Linear** in same codebase

```python
# CORRECT Layer Detection ✅
is_linear = isinstance(module, nn.Linear)
is_conv1d = type(module).__name__ == 'Conv1D'

if is_conv1d:
    # GPT-2: weight shape is (in_features, out_features)
    in_features = module.weight.shape[0]
    out_features = module.weight.shape[1]
else:
    # Standard: Linear has attributes
    in_features = module.in_features
    out_features = module.out_features
```

#### LoRA Parameter Management
- [ ] **Freeze ALL base model parameters first** (`param.requires_grad = False`)
- [ ] **Explicitly collect LoRA parameters** for optimizer (don't rely on filter)
- [ ] **LoRA parameters must be FP32** even if model is FP16
- [ ] **Move LoRA to correct device** (same as model)
- [ ] **Verify trainable parameter count** is small (0.1-3% of total)

```python
# CORRECT LoRA Injection ✅
def inject_lora(model, rank=16):
    # Step 1: Freeze EVERYTHING
    for param in model.parameters():
        param.requires_grad = False
    
    lora_params_list = []
    
    # Step 2: Add LoRA layers
    for name, module in model.named_modules():
        if is_target_layer(module):
            lora_layer = LoRALayer(in_features, out_features, rank)
            
            # Keep LoRA in FP32, even if model is FP16!
            lora_layer = lora_layer.to(
                device=module.weight.device,
                dtype=torch.float32  # Always FP32!
            )
            
            # Explicitly collect for optimizer
            lora_params_list.extend([lora_layer.lora_A, lora_layer.lora_B])
            
            module._lora = lora_layer
    
    return model, lora_params_list
```

#### LoRA Forward Pass
- [ ] **Handle dtype conversion** if model is FP16 but LoRA is FP32
- [ ] **Return to original dtype** after LoRA computation

```python
# CORRECT Mixed-Precision LoRA Forward ✅
def forward(self, x):
    original_dtype = x.dtype  # FP16 from model
    x_fp32 = x.to(torch.float32)  # Cast for FP32 LoRA
    result = x_fp32 @ self.lora_A @ self.lora_B  # Compute in FP32
    return result.to(original_dtype)  # Cast back to FP16
```

---

### 13. MARIMO VARIABLE PASSING PATTERNS

#### Underscore-Prefixed Variables
- [ ] **Underscore variables (_var) are LOCAL to a cell** - not passed to other cells
- [ ] **Remove underscore** if variable needs to be passed between cells
- [ ] **Check all return statements** to ensure critical data is returned

```python
# WRONG ❌ - Can't access _losses in other cells
@app.cell
def __(train):
    _losses = []
    for batch in data:
        _losses.append(loss)
    return _losses,  # Returns but _underscore makes it hard to use

@app.cell
def __(_losses):  # ❌ Error: _losses not defined
    plot(_losses)

# CORRECT ✅ - Remove underscore for shared variables
@app.cell
def __(train):
    training_losses = []  # No underscore
    for batch in data:
        training_losses.append(loss)
    return training_losses,

@app.cell
def __(training_losses):  # ✅ Works!
    plot(training_losses)
```

#### Cell Output Display Patterns
- [ ] **Last expression in cell is displayed** automatically
- [ ] **Expressions inside if/else are NOT displayed** (Python scoping)
- [ ] **Assign to variable, then return** for conditional display

```python
# WRONG ❌ - Nothing displays (buried in if/else)
@app.cell
def __(condition):
    if condition:
        mo.vstack([...])  # Not displayed!
    else:
        mo.md("...")  # Not displayed!

# CORRECT ✅ - Assign then return
@app.cell
def __(condition):
    if condition:
        output = mo.vstack([...])
    else:
        output = mo.md("...")
    
    output  # Top-level expression - displays!
```

#### Multi-Step Process Cells
- [ ] **Break complex operations into separate cells** for progress visibility
- [ ] **Each cell can display its own progress indicator**
- [ ] **Use `mo.stop()` at start of each cell** for conditional execution
- [ ] **Return data, not UI** from processing cells

```python
# CORRECT Multi-Step Pattern ✅
@app.cell
def __(button):
    """Step 1"""
    mo.stop(not button.value)
    result1 = process_step_1()
    mo.callout(mo.md("✅ Step 1 complete"), kind="success")
    return result1,

@app.cell
def __(button, result1):
    """Step 2"""
    mo.stop(not button.value)
    result2 = process_step_2(result1)
    mo.callout(mo.md("✅ Step 2 complete"), kind="success")
    return result2,
```

---

### 14. EDUCATIONAL DOCUMENTATION

#### "Why" Not Just "What"
- [ ] **Explain WHY decisions were made**, not just what the code does
- [ ] **Include performance implications** of design choices
- [ ] **Compare alternatives** (e.g., "Why LoRA vs full fine-tuning")
- [ ] **Explain trade-offs** (e.g., "Batch size: speed vs memory")

#### Key Topics to Explain
- [ ] **Why GPU acceleration helps** (parallelism, memory bandwidth)
- [ ] **Why specific hyperparameters** (learning rate, batch size, epochs)
- [ ] **Why this architecture** (Conv1D vs Linear, FP16 vs FP32)
- [ ] **Why this optimization** (GradScaler, gradient clipping)
- [ ] **When to use this approach** (dataset size, hardware requirements)

```markdown
# GOOD Documentation Example ✅

## Why LoRA (Low-Rank Adaptation)?

**Problem**: Full fine-tuning a 7B model requires 28GB VRAM (4 bytes × 7B params)

**Solution**: LoRA adds small trainable matrices that approximate weight updates

**Math**: 
- Original: Update W (4096 × 4096) = 16M parameters
- LoRA: Add A×B where A is (4096×16), B is (16×4096) = 131K parameters
- **Reduction**: 99% fewer parameters!

**Trade-offs**:
- ✅ 10x less memory
- ✅ 3x faster training
- ✅ Can swap adapters
- ⚠️ Slightly lower quality than full fine-tuning (95-99%)
- ⚠️ Rank choice matters (16 vs 64 can impact quality)

**When to use**:
- Limited VRAM (consumer GPUs)
- Multiple tasks (swap LoRA adapters)
- Fast iteration (train in minutes not hours)
```

#### Interactive Learning Elements
- [ ] **Dataset preview tables** so users see the data
- [ ] **Editable configuration** with clear explanations
- [ ] **Visual progress indicators** for each step
- [ ] **Result interpretation** not just raw numbers

---

### 15. DEPENDENCY AUTO-INSTALLATION PATTERNS

#### Transformers Library Pattern
- [ ] **Check if already installed** before attempting install
- [ ] **Install compatible versions** (e.g., torchvision with transformers)
- [ ] **Provide clear feedback** during installation
- [ ] **Handle import failures gracefully** with restart instructions
- [ ] **Re-import after installation** (don't rely on previous cell's import)

```python
# CORRECT Pattern ✅
@app.cell
def __(mo, subprocess):
    """Auto-install transformers"""
    transformers_available = False
    needs_restart = False
    
    # Try import first
    try:
        import transformers
        transformers_available = True
        message = mo.md("✅ Transformers already available")
    except ImportError:
        # Install if needed
        with mo.status.spinner(title="Installing transformers..."):
            result = subprocess.run(
                ["pip", "install", "--upgrade", "transformers", "torchvision"],
                capture_output=True,
                text=True,
                timeout=300
            )
        
        if result.returncode == 0:
            needs_restart = True
            message = mo.callout(
                mo.md("✅ Installed! **Please refresh page** (Cmd+R / Ctrl+R)"),
                kind="success"
            )
        else:
            message = mo.callout(
                mo.md(f"❌ Installation failed. Install manually:\n```bash\npip install transformers\n```"),
                kind="danger"
            )
    
    return transformers_available, needs_restart, message
```

---

### 16. UI CONSISTENCY

#### Color Consistency
- [ ] **All similar elements use same color** (e.g., all samples "neutral", not first "success")
- [ ] **Reserve "success"** for actual success states (completion)
- [ ] **Reserve "warn"** for warnings (not educational content)
- [ ] **Use "info"** for educational/explanatory content
- [ ] **Use "danger"** only for errors

```python
# WRONG ❌ - Inconsistent colors
for i, sample in enumerate(samples):
    mo.callout(
        sample_content,
        kind="success" if i == 0 else "neutral"  # Why is first green?
    )

# CORRECT ✅ - Consistent
for sample in samples:
    mo.callout(
        sample_content,
        kind="neutral"  # All same
    )
```

#### Table Interaction
- [ ] **Show all data with pagination** not just subset
- [ ] **Use `page_size` parameter** for reasonable page size (10-20 rows)
- [ ] **Include row indices** for reference
- [ ] **Make columns readable** (not too wide)

```python
# CORRECT Table Pattern ✅
df = pd.DataFrame({
    'Index': range(len(data)),
    'Sample': data
})

mo.ui.table(
    df,
    selection=None,  # No selection needed for preview
    page_size=10  # Show 10 rows per page
)
```

---

### 17. GPU-SPECIFIC PATTERNS FOR LLM TRAINING

#### Model-Specific Considerations
- [ ] **GPT-2 uses Conv1D** - check `type().__name__ == 'Conv1D'`
- [ ] **LLaMA uses Linear** - check `isinstance(module, nn.Linear)`
- [ ] **Different models have different layer names**:
  - GPT-2: `c_attn`, `c_proj`
  - LLaMA: `q_proj`, `v_proj`, `k_proj`, `o_proj`
  - Mistral: same as LLaMA
- [ ] **Support multiple architectures** in same code

#### Device and Dtype Placement
- [ ] **Move tensors to device immediately** when created
- [ ] **Check device of input tensors** before operations
- [ ] **Match dtypes** between model and input
- [ ] **Synchronize GPU** before timing: `torch.cuda.synchronize()`

```python
# CORRECT Device Management ✅
# Create on correct device
tensor = torch.randn(10, 10, device=device, dtype=torch.float16)

# Or move existing tensor
tensor = tensor.to(device=device, dtype=torch.float16)

# Check before operation
if tensor.device != model.device:
    tensor = tensor.to(model.device)
```

---

## Updated Testing Checklist

### LLM-Specific Tests
- [ ] Test with FP16 enabled - check for NaN loss
- [ ] Test with FP32 - should work but slower
- [ ] Verify LoRA parameter count (should be < 3% of total)
- [ ] Check gradient flow (use `requires_grad` check)
- [ ] Test text generation after training
- [ ] Verify model architecture detection (GPT-2 vs LLaMA)

### Dataset Tests
- [ ] Preview table shows all rows with pagination
- [ ] Dataset examples are clear and educational
- [ ] Generated outputs match training data patterns
- [ ] Table is scrollable and readable

### UI Tests
- [ ] All progress steps visible during training
- [ ] Colors are consistent across similar elements
- [ ] No UI elements appear then disappear unexpectedly
- [ ] Visualizations render at full width (no cutoff)

---

## Priority Additions

### CRITICAL (From LLM Fine-Tuning Experience)
1. **GradScaler for FP16** - Most common cause of NaN loss
2. **LoRA parameter collection** - Empty optimizer = training failure
3. **Variable passing** - Underscore variables not accessible
4. **Cell output display** - If/else expressions don't display

### HIGH (Significantly Improves UX)
1. **Multi-step progress** - Break complex operations into cells
2. **Educational documentation** - Explain "why" not just "what"
3. **Dataset preview** - Show data before training
4. **Architecture-specific handling** - Conv1D vs Linear

### MEDIUM (Polish)
1. **UI consistency** - Colors, layouts, interactions
2. **Auto-installation** - Handle dependencies gracefully
3. **Error messages** - Specific, actionable, helpful

---

## Common Failure Patterns (LLM Training)

### Pattern 1: NaN Loss
**Symptom**: Loss becomes NaN after a few training steps

**Causes**:
1. Missing GradScaler (FP16 training)
2. Learning rate too high
3. No gradient clipping
4. FP16 LoRA parameters (should be FP32)

**Fix Priority**:
1. Add GradScaler
2. Keep LoRA params in FP32
3. Add gradient clipping (max_norm=1.0)
4. Lower learning rate if still NaN

### Pattern 2: Empty Optimizer
**Symptom**: `ValueError: optimizer got an empty parameter list`

**Causes**:
1. Forgot to freeze base model parameters
2. LoRA layers not added correctly
3. Wrong layer name detection
4. Architecture mismatch (Conv1D vs Linear)

**Fix Priority**:
1. Freeze all params: `for p in model.parameters(): p.requires_grad = False`
2. Explicitly collect LoRA params: `trainable_params.extend([lora_A, lora_B])`
3. Check layer detection logic
4. Print parameter counts for debugging

### Pattern 3: Visualizations Not Displaying
**Symptom**: Training completes but no charts shown

**Causes**:
1. Expression inside if/else block (Python scope)
2. `mo.stop()` preventing cell execution
3. Underscore variables not passed correctly
4. Missing return statement

**Fix Priority**:
1. Assign to variable in if/else, then return at top level
2. Remove `mo.stop()` from visualization cell (use dependencies)
3. Remove underscores from variable names
4. Ensure `output` is last expression in cell

### Pattern 4: Device/Dtype Mismatch
**Symptom**: `RuntimeError: expected mat1 and mat2 to have same dtype/device`

**Causes**:
1. LoRA on CPU, model on GPU
2. LoRA in FP16, optimizer expects FP32
3. Input tensor on wrong device

**Fix Priority**:
1. Move LoRA to correct device: `lora.to(device=model.device)`
2. Keep LoRA in FP32: `lora.to(dtype=torch.float32)`
3. Handle dtype conversion in forward pass
4. Check all tensor devices before operations

---

## Integration with Existing Checklist

**Add these new sections after Section 10 (Anti-Patterns)**:
- Section 11: Mixed Precision Training
- Section 12: LoRA Implementation
- Section 13: Marimo Variable Passing
- Section 14: Educational Documentation
- Section 15: Dependency Auto-Installation
- Section 16: UI Consistency
- Section 17: GPU-Specific LLM Patterns

**Update Section 9 (Testing) to include**:
- LLM-specific tests
- Dataset preview tests
- FP16/FP32 comparison tests

**Update OUTPUT FORMAT to include new categories**:
- Educational Value score (already exists, emphasize more)
- Documentation Quality score (new emphasis)

---

**Version**: 2.0  
**Based on**: October 2025 LLM Fine-Tuning Dashboard Development  
**Key Improvements**: FP16 training, LoRA implementation, variable passing, educational focus  
**Maintained by**: Brev.dev Team

