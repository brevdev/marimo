# Single-Shot Notebook QA Prompt

**Version**: 2.0  
**Date**: October 23, 2025  
**Purpose**: Comprehensive single-shot QA for NVIDIA + Marimo notebooks  
**Based on**: Lessons from RAPIDS cuDF benchmark and LLM Fine-Tuning Dashboard development

---

## Instructions

Copy the entire "QA Checklist" section below and use it as a single prompt to QA any marimo notebook. This prompt incorporates all common issues discovered during RAPIDS development.

### Resources to Reference

Before running the QA checklist, review these resources:

1. **NVIDIA + Marimo Best Practices** (Required)
   - File: `marimo/notes/NVIDIA_MARIMO_BEST_PRACTICES.md`
   - Contains comprehensive best practices for GPU notebooks
   - Use this as the authoritative validation guide

2. **Marimo Documentation** (Reference)
   - Main docs: https://docs.marimo.io/
   - Best practices: https://docs.marimo.io/guides/best_practices/
   - Reactive execution: https://docs.marimo.io/guides/reactivity/
   - UI elements: https://docs.marimo.io/api/inputs/
   - Expensive notebooks: https://docs.marimo.io/guides/expensive_notebooks/

---

## QA Checklist

```
Please perform a comprehensive QA review of this marimo notebook.

BEFORE STARTING, review:
1. marimo/notes/NVIDIA_MARIMO_BEST_PRACTICES.md (authoritative validation guide)
2. https://docs.marimo.io/guides/best_practices/ (marimo best practices)
3. https://docs.marimo.io/guides/reactivity/ (reactive execution model)

TESTING ENVIRONMENT:
Use the Brev GPU instance for live testing: `brev shell marimo-examples-e3cd76`
Use heredoc (EOF) style for remote commands - see section 9 for examples.

Check ALL of the following:

## 1. CRITICAL MARIMO-SPECIFIC ISSUES (Blockers)

### Variable Scoping & Naming
- [ ] All variables returned from cells are actually used by dependent cells
- [ ] No variable name mismatches between cell returns and cell imports (e.g., `CUDF_AVAILABLE` vs `cudf_available`)
- [ ] Consistent naming across all cells (check uppercase vs lowercase, underscores)
- [ ] No duplicate variable names defined in multiple cells (marimo will error)
- [ ] All UI elements are returned from the cell where they're defined
- [ ] UI element `.value` is accessed correctly in dependent cells

### Return Statement Placement
- [ ] NO return statements inside if/else blocks or conditionals
- [ ] All cells have a single implicit return at the very end (or explicit `return var1, var2`)
- [ ] No early returns that would cause `SyntaxError: 'return' outside function`
- [ ] Output accumulated into variables (e.g., `_output`) before final return if using conditionals

### Module Import Issues
- [ ] Modules that might fail import (cudf, dask, etc.) are re-imported fresh in installation cells
- [ ] Don't rely on failed imports from previous cells (e.g., if Cell-1 sets `cudf = None`, Cell-2 must re-import)
- [ ] All modules passed between cells are actually the module object, not None
- [ ] Check that `import X as Y` actually assigns Y correctly after installation

### Reactive Execution
- [ ] NO use of `on_change` handlers (use reactive cell dependencies instead)
- [ ] `mo.stop()` used correctly with boolean condition as first arg
- [ ] Expensive operations gated by `mo.ui.run_button()` or similar
- [ ] Cell dependencies are correctly declared in function signature

## 2. GPU CODE ISSUES

### Memory Management
- [ ] Large tensors/dataframes explicitly deleted with `del variable`
- [ ] `torch.cuda.empty_cache()` called after deleting large GPU objects
- [ ] Try-except for `torch.cuda.OutOfMemoryError` with helpful error messages
- [ ] Memory cleanup happens even if operations fail (use try-finally if needed)
- [ ] No memory leaks from variables left in global scope

### GPU Detection & Validation
- [ ] Check `torch.cuda.is_available()` before any GPU operations
- [ ] Use `mo.stop()` with clear error message if no GPU detected
- [ ] Display GPU name, memory, compute capability to user
- [ ] Handle multi-GPU scenarios (enumerate all devices, document which is used)

### GPU Operations
- [ ] `torch.cuda.synchronize()` called before timing measurements
- [ ] Warmup iterations before benchmarking
- [ ] Random seeds set for reproducibility (`torch.manual_seed`, `np.random.seed`)

## 3. ERROR HANDLING & EDGE CASES

### None Value Handling
- [ ] All calculations check for None values (e.g., `[x for x in list if x is not None]`)
- [ ] Division operations check for empty lists or zero denominators
- [ ] Speedup calculations handle cases where CPU or GPU didn't run
- [ ] Filter None from lists before calling `sum()`, `max()`, `min()`, `mean()`

### Error Messages
- [ ] All try-except blocks have informative error messages
- [ ] Error messages suggest solutions, not just state the problem
- [ ] Use `mo.callout(kind="danger")` for errors, "warn" for warnings
- [ ] Include current state in error messages (e.g., available GPU memory)

### Dependency Handling
- [ ] Missing optional dependencies have graceful fallbacks
- [ ] Installation attempts wrapped in try-except
- [ ] Clear messaging about what features require which dependencies

## 4. PROGRESS INDICATORS & USER FEEDBACK

### Spinner Usage
- [ ] `mo.status.spinner()` used AS A CONTEXT MANAGER with `with` statement
- [ ] Never try to call `.open()` or `.close()` on spinner (doesn't exist)
- [ ] Spinner has descriptive `title` and `subtitle` parameters
- [ ] Spinner wraps the entire long-running operation

```python
# CORRECT ✅
with mo.status.spinner(title="Running...", subtitle="This may take a minute"):
    result = expensive_operation()

# WRONG ❌
spinner = mo.status.spinner(title="Running...")
spinner.open()  # This will fail - no .open() method
result = expensive_operation()
spinner.close()  # This will fail - no .close() method
```

### User Feedback
- [ ] Progress messages printed during long operations
- [ ] Results displayed with context and interpretation
- [ ] Success/failure clearly indicated with callouts
- [ ] Performance metrics displayed (time, speedup, throughput)

## 5. UI/UX ISSUES

### Mode Detection & Dropdowns
- [ ] Dropdown `.value` returns the selected KEY, but might return LABEL in some cases
- [ ] Mode detection checks for both keys and labels if dropdown behavior is inconsistent
- [ ] Use normalized boolean flags for mode checks instead of raw strings

```python
# ROBUST ✅
_run_cpu = mode_toggle.value in ['cpu', 'both', 'CPU Only', 'CPU vs GPU']
_run_gpu = mode_toggle.value in ['gpu', 'both', 'GPU Only', 'CPU vs GPU']

# FRAGILE ❌
if mode_toggle.value == 'both':  # Might break if dropdown returns label
    run_benchmark()
```

### Chart Rendering
- [ ] Plotly charts have adequate `height` (400-600px minimum)
- [ ] Charts have `margin=dict(t=60, l=80, r=40, b=80)` to prevent label cutoff
- [ ] Interactive features enabled (hover, zoom, pan)
- [ ] Axes labeled clearly
- [ ] Chart titles are descriptive

### Layout
- [ ] Use `mo.vstack()` and `mo.hstack()` for organized layouts
- [ ] Related UI elements grouped together
- [ ] Callouts used for important information
- [ ] Consistent emoji usage (not excessive)

## 6. CODE QUALITY ISSUES

### String Formatting
- [ ] All f-strings have the `f` prefix (easy to miss!)
- [ ] F-string expressions are valid (check `.0f`, `.2f` formatting)
- [ ] No raw strings where interpolation was intended

```python
# WRONG ❌ - Missing f prefix
message = "GPU usage: {util:.0f}%"  # Prints literal string

# CORRECT ✅
message = f"GPU usage: {util:.0f}%"  # Interpolates value
```

### Type Consistency
- [ ] Function parameters have type hints
- [ ] Return types documented
- [ ] Dict/List keys are consistent types (don't mix str and int)
- [ ] Value types match expected types in calculations

### Function Encapsulation
- [ ] Complex logic encapsulated in functions
- [ ] Functions have docstrings
- [ ] Temporary variables prefixed with `_` to avoid namespace pollution
- [ ] Only necessary variables returned from cells

## 7. DOCUMENTATION

### Module Docstring
- [ ] Clear title describing notebook purpose
- [ ] 2-3 sentence description
- [ ] Feature list (bullet points)
- [ ] GPU requirements (VRAM, compute capability)
- [ ] List of tested hardware configurations
- [ ] CUDA/driver version requirements
- [ ] Author and date

### Cell Documentation
- [ ] Every cell has a descriptive docstring
- [ ] Complex operations have inline comments explaining WHY
- [ ] Function docstrings include Args and Returns
- [ ] Links to relevant documentation where appropriate

## 8. PERFORMANCE & OPTIMIZATION

### Benchmarking
- [ ] Warmup iterations before timing
- [ ] `torch.cuda.synchronize()` before timing GPU operations
- [ ] Multiple iterations for stable measurements
- [ ] Throughput and speedup calculations are correct
- [ ] Results include statistical measures (mean, std, min, max)

### Data Generation
- [ ] Random seeds set for reproducibility
- [ ] Dataset sizes scale with available GPU memory
- [ ] Data generation is reasonably fast
- [ ] Memory usage is reasonable for target GPUs

### Visualization
- [ ] Charts render quickly
- [ ] No unnecessary re-computation
- [ ] Caching used for expensive operations (`@mo.cache` if applicable)

## 9. EXECUTION FLOW TESTING

### Testing Environment

**Use the Brev test instance for live validation**:
```bash
brev shell marimo-examples-e3cd76
```

Use heredoc (EOF style) to run commands remotely:
```bash
brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo
git pull origin main
marimo edit notebook_name.py
EOF
```

### Manual Testing Checklist
- [ ] Cell 1 (imports) executes without errors
- [ ] GPU detection cell shows correct GPU info
- [ ] UI controls render and are interactive
- [ ] Clicking buttons triggers expected behavior
- [ ] Long operations show progress indicator
- [ ] Results display correctly after completion
- [ ] Charts are interactive (hover, zoom, pan)
- [ ] Error cases handled gracefully (try invalid inputs)
- [ ] Notebook can run multiple times without restart

### Testing Commands for Brev Instance

**Deploy notebook to test instance**:
```bash
brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo
git pull origin main

# Check notebook syntax
python -m py_compile notebook_name.py

# Run headless execution test
marimo run notebook_name.py --headless

echo ""
echo "✅ Syntax check complete"
echo "To open interactively: marimo edit notebook_name.py"
EOF
```

**Test GPU operations**:
```bash
brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo

# Verify GPU is available
nvidia-smi

# Check CUDA toolkit
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else \"N/A\"}')"

# Test notebook GPU detection
marimo run notebook_name.py --headless
EOF
```

**Check for common errors**:
```bash
brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo

# Check for Python syntax errors
python -m py_compile notebook_name.py

# Check for undefined variables (basic lint)
python3 << 'PYTHON_EOF'
import ast
import sys

with open('notebook_name.py', 'r') as f:
    try:
        ast.parse(f.read())
        print("✅ No syntax errors")
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        sys.exit(1)
PYTHON_EOF

echo ""
echo "Syntax validation complete"
EOF
```

**Interactive testing session**:
```bash
# Open notebook in browser for manual testing
brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo
echo ""
echo "Starting marimo server..."
echo "Notebook will be available at the forwarded port"
echo ""
marimo edit notebook_name.py --host 0.0.0.0 --port 8080
EOF
```

### Edge Cases
- [ ] Test with minimal dataset size
- [ ] Test with maximum dataset size
- [ ] Test with CPU-only mode (if applicable)
- [ ] Test with GPU-only mode (if applicable)
- [ ] Test with missing optional dependencies
- [ ] Test with all operations selected
- [ ] Test with single operation selected

### Testing Best Practices

1. **Always pull latest code** before testing
2. **Check syntax first** with `python -m py_compile`
3. **Test headless** with `marimo run --headless` to catch execution errors
4. **Test interactively** with `marimo edit` for UI/UX validation
5. **Verify GPU detection** with `nvidia-smi` and torch checks
6. **Test edge cases** systematically (min/max values, missing deps)
7. **Document test results** with screenshots or error logs

## 10. SPECIFIC ANTI-PATTERNS TO CHECK

### ❌ WRONG: Return in conditional
```python
@app.cell
def __(condition):
    if condition:
        result = "yes"
        return result  # ❌ WRONG - breaks marimo
    else:
        result = "no"
        return result  # ❌ WRONG - breaks marimo
```

### ✅ CORRECT: Single return at end
```python
@app.cell
def __(condition):
    if condition:
        result = "yes"
    else:
        result = "no"
    return result,  # ✅ CORRECT - single return
```

### ❌ WRONG: Variable name mismatch
```python
@app.cell
def __():
    CUDF_AVAILABLE = True
    return CUDF_AVAILABLE,

@app.cell
def __(cudf_available):  # ❌ WRONG - name doesn't match
    if cudf_available:
        pass
```

### ✅ CORRECT: Consistent naming
```python
@app.cell
def __():
    cudf_available = True
    return cudf_available,

@app.cell
def __(cudf_available):  # ✅ CORRECT - matches exactly
    if cudf_available:
        pass
```

### ❌ WRONG: Calling spinner methods
```python
@app.cell
def __(mo):
    spinner = mo.status.spinner(title="Working...")
    spinner.open()  # ❌ AttributeError: no .open() method
    result = work()
    spinner.close()  # ❌ AttributeError: no .close() method
    return result,
```

### ✅ CORRECT: Spinner as context manager
```python
@app.cell
def __(mo):
    with mo.status.spinner(title="Working..."):  # ✅ CORRECT
        result = work()
    return result,
```

### ❌ WRONG: Not filtering None before calculations
```python
@app.cell
def __(results):
    # results['times'] might contain None values
    avg_time = sum(results['times']) / len(results['times'])  # ❌ TypeError
    return avg_time,
```

### ✅ CORRECT: Filter None values
```python
@app.cell
def __(results):
    valid_times = [t for t in results['times'] if t is not None]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0  # ✅ CORRECT
    return avg_time,
```

### ❌ WRONG: Missing f-string prefix
```python
@app.cell
def __(speedup):
    message = "Speedup: {speedup:.1f}x"  # ❌ Prints literal string
    return message,
```

### ✅ CORRECT: F-string with prefix
```python
@app.cell
def __(speedup):
    message = f"Speedup: {speedup:.1f}x"  # ✅ CORRECT - interpolates
    return message,
```

---

## OUTPUT FORMAT

For each issue found, provide:
1. **Location**: Cell number or line number
2. **Issue**: Brief description
3. **Best Practice Reference**: Which section of NVIDIA_MARIMO_BEST_PRACTICES.md this violates (if applicable)
4. **Severity**: CRITICAL (blocks execution) | HIGH (causes errors) | MEDIUM (poor UX) | LOW (polish)
5. **Fix**: Specific code change needed

If no issues found, state: "✅ All checks passed - notebook is production ready"

At the end of the review, provide a summary score matching the format in NVIDIA_MARIMO_BEST_PRACTICES.md:
- Documentation & Structure: X/10
- Code Organization: X/10
- GPU Resource Management: X/10
- Reactivity & Interactivity: X/10
- Performance & Optimization: X/10
- Error Handling: X/10
- Reproducibility: X/10
- User Experience: X/10
- Testing & Validation: X/10
- Educational Value: X/10
- **TOTAL**: X/100

---

## PRIORITY LEVELS

### CRITICAL (Must Fix Before Production)
- Marimo variable scoping errors
- Return statement placement errors
- GPU OOM without handling
- Missing progress indicators for long operations
- Broken execution flow

### HIGH (Fix Before Release)
- None value handling in calculations
- Missing error messages
- Poor user feedback
- Chart rendering issues
- Documentation gaps

### MEDIUM (Fix Soon)
- Suboptimal layouts
- Missing edge case handling
- Performance optimizations
- Code organization improvements

### LOW (Nice to Have)
- Polish and styling
- Additional features
- Enhanced documentation
- Optional validations

---

## 11. MIXED PRECISION TRAINING (FP16/BF16)

### GradScaler Requirements
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

### Dtype Management
- [ ] **Check that model and data dtypes match** during forward pass
- [ ] **Handle dtype conversion** in custom layers if mixing FP16 model with FP32 parameters
- [ ] **Test for NaN loss** - primary symptom of FP16 issues
- [ ] **Gradient clipping** is essential with FP16 (prevents NaN)

### Common FP16 Errors
- [ ] `ValueError: Attempting to unscale FP16 gradients` → trainable params must be FP32
- [ ] `RuntimeError: expected same dtype` → model FP16 but params FP32, need conversion layer
- [ ] NaN loss → missing GradScaler or gradient clipping

## 12. LORA IMPLEMENTATION

### Architecture-Specific Layer Detection
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

### LoRA Parameter Management
- [ ] **Freeze ALL base model parameters first** (`param.requires_grad = False`)
- [ ] **Explicitly collect LoRA parameters** for optimizer (don't rely on filter)
- [ ] **LoRA parameters must be FP32** even if model is FP16
- [ ] **Move LoRA to correct device** (same as model)
- [ ] **Verify trainable parameter count** is small (0.1-3% of total)

### LoRA Forward Pass
- [ ] **Handle dtype conversion** if model is FP16 but LoRA is FP32
- [ ] **Return to original dtype** after LoRA computation

## 13. MARIMO VARIABLE PASSING PATTERNS

### Underscore-Prefixed Variables
- [ ] **Underscore variables (_var) are LOCAL to a cell** - not passed to other cells
- [ ] **Remove underscore** if variable needs to be passed between cells
- [ ] **Check all return statements** to ensure critical data is returned

### Cell Output Display Patterns
- [ ] **Last expression in cell is displayed** automatically
- [ ] **Expressions inside if/else are NOT displayed** (Python scoping)
- [ ] **Assign to variable, then return** for conditional display

### Multi-Step Process Cells
- [ ] **Break complex operations into separate cells** for progress visibility
- [ ] **Each cell can display its own progress indicator**
- [ ] **Use `mo.stop()` at start of each cell** for conditional execution
- [ ] **Return data, not UI** from processing cells

## 14. EDUCATIONAL DOCUMENTATION

### "Why" Not Just "What"
- [ ] **Explain WHY decisions were made**, not just what the code does
- [ ] **Include performance implications** of design choices
- [ ] **Compare alternatives** (e.g., "Why LoRA vs full fine-tuning")
- [ ] **Explain trade-offs** (e.g., "Batch size: speed vs memory")

### Key Topics to Explain
- [ ] **Why GPU acceleration helps** (parallelism, memory bandwidth)
- [ ] **Why specific hyperparameters** (learning rate, batch size, epochs)
- [ ] **Why this architecture** (Conv1D vs Linear, FP16 vs FP32)
- [ ] **Why this optimization** (GradScaler, gradient clipping)
- [ ] **When to use this approach** (dataset size, hardware requirements)

### Interactive Learning Elements
- [ ] **Dataset preview tables** so users see the data
- [ ] **Editable configuration** with clear explanations
- [ ] **Visual progress indicators** for each step
- [ ] **Result interpretation** not just raw numbers

## 15. DEPENDENCY AUTO-INSTALLATION PATTERNS

### Transformers Library Pattern
- [ ] **Check if already installed** before attempting install
- [ ] **Install compatible versions** (e.g., torchvision with transformers)
- [ ] **Provide clear feedback** during installation
- [ ] **Handle import failures gracefully** with restart instructions
- [ ] **Re-import after installation** (don't rely on previous cell's import)

## 16. UI CONSISTENCY

### Color Consistency
- [ ] **All similar elements use same color** (e.g., all samples "neutral", not first "success")
- [ ] **Reserve "success"** for actual success states (completion)
- [ ] **Reserve "warn"** for warnings (not educational content)
- [ ] **Use "info"** for educational/explanatory content
- [ ] **Use "danger"** only for errors

### Table Interaction
- [ ] **Show all data with pagination** not just subset
- [ ] **Use `page_size` parameter** for reasonable page size (10-20 rows)
- [ ] **Include row indices** for reference
- [ ] **Make columns readable** (not too wide)

## 17. GPU-SPECIFIC LLM PATTERNS

### Model-Specific Considerations
- [ ] **GPT-2 uses Conv1D** - check `type().__name__ == 'Conv1D'`
- [ ] **LLaMA uses Linear** - check `isinstance(module, nn.Linear)`
- [ ] **Different models have different layer names**:
  - GPT-2: `c_attn`, `c_proj`
  - LLaMA: `q_proj`, `v_proj`, `k_proj`, `o_proj`
  - Mistral: same as LLaMA
- [ ] **Support multiple architectures** in same code

### Device and Dtype Placement
- [ ] **Move tensors to device immediately** when created
- [ ] **Check device of input tensors** before operations
- [ ] **Match dtypes** between model and input
- [ ] **Synchronize GPU** before timing: `torch.cuda.synchronize()`

---

## COMMON FAILURE PATTERNS (NEW)

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

END OF QA CHECKLIST
```

---

## Usage Example

1. **Copy the QA Checklist** section above
2. **Add the notebook file** to the prompt
3. **Run the QA** with your AI assistant
4. **Address issues** by priority level
5. **Deploy to Brev instance** for live testing
6. **Re-test** after fixes

## Quick Test Commands

### Local Testing
```bash
# Check for Python syntax errors
python -m py_compile notebook.py

# Test notebook execution (headless)
marimo run notebook.py

# Open notebook for interactive testing
marimo edit notebook.py
```

### Brev Instance Testing (Recommended)
```bash
# Deploy and test on real GPU hardware
brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo
git pull origin main

# Syntax check
python -m py_compile notebook_name.py

# GPU verification
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else \"N/A\"}')"

# Headless execution test
marimo run notebook_name.py --headless

echo ""
echo "✅ Testing complete"
echo "For interactive testing: marimo edit notebook_name.py --host 0.0.0.0 --port 8080"
EOF
```

### After Fixes - Quick Validation
```bash
# Push changes and validate on Brev
git add marimo/notebook_name.py
git commit -m "Fix: <description>"
git push origin main

brev shell marimo-examples-e3cd76 << 'EOF'
cd ~/marimo
git pull origin main
marimo run notebook_name.py --headless && echo "✅ Validation passed" || echo "❌ Validation failed"
EOF
```

---

## Known Patterns from RAPIDS Development

### Pattern 1: Module Import After Installation
```python
@app.cell
def __():
    """Initial import attempt"""
    try:
        import cudf
        CUDF_AVAILABLE = True
    except ImportError:
        CUDF_AVAILABLE = False
        cudf = None
    return cudf, CUDF_AVAILABLE

@app.cell
def __(CUDF_AVAILABLE):
    """Installation cell - must re-import!"""
    cudf_module = None
    cudf_available = False
    
    # Always try fresh import first
    try:
        import cudf as cudf_module
        cudf_available = True
        print("✅ cuDF available")
    except ImportError:
        print("⚠️ Installing cuDF...")
        # ... installation logic ...
        # Re-import after installation
        try:
            import cudf as cudf_module
            cudf_available = True
        except ImportError:
            cudf_available = False
    
    return cudf_module, cudf_available  # Pass fresh module
```

### Pattern 2: Robust Mode Detection
```python
@app.cell
def __(mode_toggle):
    """Normalize mode detection"""
    # Dropdown might return key OR label
    _mode_value = mode_toggle.value
    
    # Normalize to boolean flags
    _run_cpu = _mode_value in ['cpu', 'both', 'CPU Only', 'CPU vs GPU']
    _run_gpu = _mode_value in ['gpu', 'both', 'GPU Only', 'CPU vs GPU']
    
    return _run_cpu, _run_gpu
```

### Pattern 3: Safe Calculations
```python
@app.cell
def __(results):
    """Calculate metrics safely"""
    # Filter None values
    valid_times = [t for t in results['times'] if t is not None]
    valid_speedups = [s for s in results['speedup'] if s is not None]
    
    # Check for empty lists
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
    max_speedup = max(valid_speedups) if valid_speedups else 1.0
    
    return avg_time, max_speedup
```

### Pattern 4: Spinner Pattern
```python
@app.cell
def __(mo, run_button):
    """Long-running operation with spinner"""
    mo.stop(not run_button.value, mo.md("Click button to start"))
    
    # Spinner as context manager
    with mo.status.spinner(
        title="Running benchmark...",
        subtitle="This may take 30-60 seconds"
    ):
        results = expensive_benchmark()
    
    return results,
```

---

## Validation Resources

### Primary Validation Guide

**NVIDIA + Marimo Best Practices** (`marimo/notes/NVIDIA_MARIMO_BEST_PRACTICES.md`)
- 10-category validation framework
- Comprehensive validation checklist
- Common pitfalls to avoid
- Quick reference card
- Score: X/100 format

Use this as the authoritative reference for:
- Documentation requirements
- Code organization standards
- GPU resource management patterns
- Marimo reactivity best practices
- UI/UX guidelines
- Error handling patterns
- Performance optimization
- Educational value assessment

### Marimo Official Documentation

**Core Guides**:
- [Best Practices](https://docs.marimo.io/guides/best_practices/) - Coding patterns and anti-patterns
- [Reactive Execution](https://docs.marimo.io/guides/reactivity/) - How marimo's reactive execution works
- [Expensive Notebooks](https://docs.marimo.io/guides/expensive_notebooks/) - Handling long-running operations
- [UI Elements](https://docs.marimo.io/api/inputs/) - Complete UI element reference

**Key Marimo Principles**:
1. **Reactive execution**: Cells re-run when dependencies change
2. **No callbacks**: Don't use `on_change` handlers
3. **Single return**: Return statement only at end of cell
4. **Clean namespace**: Minimize global variables
5. **Explicit dependencies**: Declare all dependencies in function signature

### NVIDIA GPU Documentation

**Reference Links**:
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [TensorRT Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [RAPIDS Documentation](https://docs.rapids.ai/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)

---

## Changelog

- **v2.0** (2025-10-23): Major expansion based on LLM Fine-Tuning Dashboard development
  - **Added 7 new sections** (11-17):
    - Section 11: Mixed Precision Training (FP16/BF16)
    - Section 12: LoRA Implementation
    - Section 13: Marimo Variable Passing Patterns
    - Section 14: Educational Documentation
    - Section 15: Dependency Auto-Installation Patterns
    - Section 16: UI Consistency
    - Section 17: GPU-Specific LLM Patterns
  - **Added 4 Common Failure Patterns**:
    - NaN Loss (GradScaler, learning rate, clipping)
    - Empty Optimizer (parameter freezing, LoRA collection)
    - Visualizations Not Displaying (Python scope, mo.stop())
    - Device/Dtype Mismatch (FP16/FP32, CPU/GPU)
  - **Key learnings from 16 bugs fixed**:
    - FP16 training requires GradScaler + FP32 trainable params
    - GPT-2 Conv1D has opposite weight shape from nn.Linear
    - Marimo underscore variables are local to cells
    - Educational documentation is critical for user understanding
  - **Validation points increased from ~100 to 150+**
  
- **v1.0** (2025-10-23): Initial version based on RAPIDS development lessons
  - Added references to NVIDIA_MARIMO_BEST_PRACTICES.md
  - Added marimo documentation links
  - Added scoring format to match best practices guide
  - Added Brev instance testing section with heredoc (EOF) examples
  - Added comprehensive testing commands for GPU validation
  - Added testing best practices and workflow

---

**Maintained by**: Brev.dev Team  
**Last Updated**: October 23, 2025

