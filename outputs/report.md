# NimbusAI - GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Root Cause Analysis: The GPU-Util Lie

- **The Illusion:** Traditional monitoring via `nvidia-smi` reports `gpu_util_pct` based on sampling whether the GPU clock was active at all during a 1-second interval. It does **not** measure mathematical throughput or compute utilization.
- **The Reality (MFU vs GPU-Util):** `gpu-h100-4` reports **98.0% GPU-Util** but achieves an MFU (Model FLOPs Utilization) of only **0.200 (20.0%)**. Similarly, `gpu-a10g-1` shows **97.0% GPU-Util** with only **0.270 MFU**.
- **Root Cause:** When workloads are memory-bound (e.g., autoregressive LLM token decoding with small batch sizes) or suffer from non-fused kernels and kernel launch latency, Tensor Cores spend up to 80% of clock cycles stalled waiting for high-bandwidth memory (HBM) data transfers.
- **Financial Impact:** NimbusAI is paying full on-demand rates ($2.50/hr for H100) for a fifth of the compute capability. Right-sizing `gpu-h100-4` to an A100 saves **$655/month** without degrading application throughput.

## Strategic Action Plan (Prioritized by ROI)

1. **Purchasing Optimization (Priority #1 - ROI: $10,040/month, 80.3% of total savings):**
   - Commit 24/7 inference workloads (`job-infer-chat`, `job-infer-rag`) to 3-year Reserved Instances (break-even utilization is 55% at 45% discount; continuous workloads operate at 100% duty cycle).
   - Shift interruptible training jobs (`job-train-llm`, `job-train-embed`, `job-finetune`, `job-dev-sandbox`, `job-batch-eval`) to Spot Instances with automated checkpointing, achieving 39.1% net purchasing savings after accounting for checkpoint and rework overhead.
2. **Inference Software Levers (Priority #2 - ROI: $1,212/month, cuts unit cost from $6.488 to $1.126/1M tokens, -82.6%):**
   - **Cascade Routing:** Direct ~80% of low-complexity requests to small models ($0.20/1M in vs $3.00/1M in).
   - **Prompt Caching:** Apply 90% discount on cached repetitive system prompts across chat and RAG services.
   - **Batch API:** Process offline evaluation queries asynchronously at 50% discount.
3. **Right-Size Over-Provisioned GPUs (Priority #3 - ROI: $655/month):**
   - Downgrade identified GPU-Util lie instances (`gpu-h100-4`) to appropriately matched tiers (A100).
4. **Automate Idle GPU Reclamation (Priority #4 - ROI: $600/month):**
   - Enforce auto-shutdown policies for training GPUs idle overnight (<10% utilization for >1 hour, e.g. `gpu-h100-5`).
5. **Governance & Chargeback Readiness (Priority #5):**
   - Current tag coverage reached **92%**, successfully clearing the 80% chargeback readiness threshold. Enable monthly cost chargebacks across `assistant`, `rag`, `search`, and `eval` departments.

## Sustainability & Regional Grid Carbon Analysis

Grid carbon intensity directly dictates environmental footprint, while electricity tariffs dictate base infrastructure cost:

| Region | Grid Carbon (gCO2/kWh) | Electricity ($/kWh) | Monthly Carbon (kg CO2e) | Energy Cost ($) | Profile |
|---|---|---|---|---|---|
| `europe-north1` | 30 | $0.090 | 126.8 | $380.4 | Cleanest (Norway Hydro) |
| `us-east-wa` | 90 | $0.055 | 380.4 | $232.5 | Cheapest Electricity (WA Hydro) |
| `us-west-2` | 120 | $0.070 | 507.2 | $295.9 | Balanced (Pacific NW) |
| `us-east-1` | 380 | $0.120 | 1,606.3 | $507.2 | Baseline Default (Virginia) |
| `europe-central2` | 660 | $0.180 | 2,789.8 | $760.9 | Carbon-Intensive (Poland Coal) |

- **Carbon-Electricity Link:** Hydropower-rich regions provide both superior green credentials (30-90 gCO2/kWh) and lower industrial electricity rates ($0.055-$0.09/kWh) compared to coal/gas regions ($0.12-$0.18/kWh).
- **Latency Trade-off:** Asynchronous batch training and evaluation have zero interactive latency sensitivity. Moving interruptible workloads to `europe-north1` cuts **1,479.5 kg CO2e/month (92.1% reduction)** with zero user impact.

## Advanced Extensions Implemented (Your Turn)

### Extension 3: Prompt Cache Economics (`cache_is_worth_it`)
- **Economic Model:** Prompt caching requires cache write/storage investment. It is economically viable only when: `avg_reads * (1 - read_discount) * read_price >= write_cost`.
- **Break-Even Threshold:** At 90% read discount (`read_discount=0.10`), break-even read count is **1.11 reads** for both small ($0.20/1M) and large ($3.00/1M) models.
- **Observed Traffic:** In `token_usage.csv`, shared system prompts across chat/RAG average **12.0 reads** per cached prefix. The condition `cache_is_worth_it()` evaluates to **True**, mathematically validating the prompt caching lever.

### Extension 4: Reasoning Traffic & Energy Budget
- **Traffic Disparity:** Reasoning queries account for only **8.4% of requests** (201/2,400) and **16.5% of tokens** (1.24M/7.53M), but consume **94.0% of total inference energy** (29,787.7 Wh out of 31,675.3 Wh).
- **Energy Intensity:** Reasoning averages **148.20 Wh per query** vs **0.86 Wh** for standard queries (**172.6x higher**), driven by combined long chain-of-thought generation (6x tokens) and complex step expansion (80x energy multiplier).
- **Proposed Routing Policy:** Gate reasoning behind a task difficulty classifier; only route to reasoning models when prompt complexity score > 0.75. Capping reasoning traffic to 4% (50% reduction) saves **14.8 kWh/day (444.6 kWh/month)**.

### Extension 5: Carbon-Aware Workload Scheduling
- **Workload Scope:** Evaluated 5 interruptible jobs totaling 4,227.0 kWh/month of GPU computation.
- **Emissions Comparison:** Running in `us-east-1` produces **1,606.3 kg CO2e/month**, whereas running in `europe-north1` produces only **126.8 kg CO2e/month**.
- **Environmental Savings:** Shifting interruptible jobs saves **1,479.5 kg CO2e/month (92.1% reduction)** while saving **$126.81/month** in electricity bills.

_Figures are June-2026 as-of snapshots; re-baseline before acting._