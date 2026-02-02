# GPT-2 Benchmark (BottleCapAI) 

The following is an overview of ideas and experimental choices to reduce the training time of a GPT-2 model without sacrificing performance. The experiments were run on a single NVIDIA RTX 4090.

## Results

The reported baseline values come from a re-run of the original code to ensure a fair comparison on the same hardware. The implemented improvements offer ~5% faster training, saving over 16 minutes of training time.

| Metric | Baseline | Submission | Change |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | 3.384 | **3.382** | -0.002 |
| **Training Time** | 20464.66 s (~5.68 hr) | **19502.80 s** (~5.41 hr) | -16m 02s |
| **Avg Step Duration** | 4292.09 ms | **4090.35 ms** | 4.7% faster |
| **Throughput** | ~122k tok/sec | **~128k tok/sec** | 1.05x |

## What Worked (Speed-ups)

The following design choices were utilized to speed up the run compared to the baseline.

1. **Increasing vocab size from 50257 to 50304.**
   A simple hardware optimization "trick". NVIDIA GPUs compute matrix multiplications most efficiently when the dimensions are multiples of specific powers of 2 (e.g. 64 or 128). Matrices with "odd" shapes cannot be efficiently tiled by the GPU, which slows down processing. 
   
   Padding the vocab size causes a slight increase in computation (predicting padding tokens), but removing the "computational friction" of unaligned memory access makes it worth it.

2. **GELU approximation.**
   Replacing the default `torch` GELU activation with a `tanh` approximation. This replaces the expensive error function (`erf`) calculations with basic multiplications and additions, skipping the heavy iterative logic required by the exact GELU.
   
   The loss of precision is negligible for convergence but offers a speedup.

3. **Fused AdamW.**
   The fused version of the optimizer avoids repeated trips to memory, resulting in better GPU utilization (specifically for NVIDIA GPUs).

4. **Pinned memory and non-blocking transfer.**
   Enabled `pin_memory=True` and `non_blocking=True` to eliminate CPU overhead and allow data transfer from RAM to VRAM to overlap with computation.

## What Didn't Work (Negative Results)

The following design choices were tested but eventually discarded because they either slowed down training or prevented fast convergence. (Due to computational restrictions, experiments with these options were usually terminated ~1000 steps into training if deemed inefficient.)

1. **Curriculum learning.**
   The idea was to speed up convergence by training on shorter sequences with larger batch sizes in the beginning, then gradually increasing length and complexity. In theory, the attention mechanism is cheaper on shorter strings. 
   However, because the model is compiled with `torch.compile()`, changing the context length triggers a graph recompilation at every step of the curriculum. This introduced a compilation overhead that outweighed the potential speedup.

2. **Replacing standard MLP with SwiGLU.**
   SwiGLU is generally considered a "smarter" alternative to standard FFNs. However, because it is a gated unit, it requires 3 matrix multiplications instead of 2.
   
   While parameter counts can be adjusted to match the original MLP, this reduces the capacity of the network. From empirical evidence, the training and validation losses didn't converge as fast as the baseline, negating the potential improvement.

3. **QK-Norm.**
   Applying LayerNorm to the Query (Q) and Key (K) vectors prevents "logit drift", where the magnitude of Q and K grows significantly, leading to saturated Softmax values. While QK-Norm allows for larger learning rates, the two additional LayerNorm operations in Attention slowed down step times enough to offset the benefits of the higher learning rate.

4. **GPT-J trick (Parallel Blocks).**
   An architectural change that replaces serial Attention and FFN execution with a parallel one. This unlocks a specific optimization called "Kernel Fusion for Projections."
   
   However, in this parallel version, the FFN is blind to the Attention's result for that specific layer. This degradation of information flow presumably hurt the convergence speed.

5. **Differential Attention.**
   The goal was to focus attention on important tokens and filter out "background" noise. However, the operator is incompatible with standard FlashAttention kernels. Falling back to a slower attention implementation increased clock time significantly, making it unsuitable for this challenge.
  