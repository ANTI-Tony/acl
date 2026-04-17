# Math Instruction Data Optimization

基于 SR-IOID 框架的数学推理方向拓展，针对数学指令数据进行自动化质量优化。

## Framework

```
Raw Math Data (5K)
    │
    ▼
[IQD] Instruction Quality Differentiation
    │   - 数学5维质量评估（计算正确性/推理完整性/逻辑一致性/问题清晰度/策略合理性）
    │   - K-means 语义聚类 + IFD 难度打分
    │
    ├── High-quality data ──────────────────┐
    │                                       │
    ▼                                       │
[FIR] Feedback-driven Iterative Refinement  │
    │   - Step-level verification           │
    │   - 结构化数学反馈                      │
    │   - 评估→修正→审核 (T=3)               │
    │                                       │
    ▼                                       │
[OA] Output Alignment                      │
    │   - 数值一致性对齐                      │
    │   - 符号/变量统一性检查                  │
    │                                       │
    ▼                                       ▼
    Merge ──────────────────────────► Optimized Dataset
                                            │
                                            ▼
                                    Fine-tune Qwen2.5-3B
                                            │
                                            ▼
                                    Evaluate on 6 Benchmarks
```

## Benchmarks

| Benchmark | Description |
|-----------|-------------|
| MATH500 | MATH test subset (500) |
| Min. Math | Minerva Math |
| Olympiad | OlympiadBench |
| Col. Math | College Math |
| AMC23 | AMC 2023 |
| AIME25 | AIME 2025 |

## Hardware

- 2× NVIDIA A100 80GB
- Expert model: Qwen2.5-72B-Instruct (4bit quantized, vLLM)
- Backbone model: Qwen2.5-3B (LoRA fine-tuning)

## Usage

```bash
# Step 1: Download and sample data
python scripts/prepare_data.py

# Step 2: Deploy expert model
bash scripts/serve_expert.sh

# Step 3: Run data optimization pipeline
python scripts/run_iqd.py          # Quality differentiation
python scripts/run_fir.py          # Iterative refinement
python scripts/run_oa.py           # Output alignment

# Step 4: Fine-tune backbone
bash scripts/train.sh

# Step 5: Evaluate
bash scripts/eval.sh

# Step 6: Bad case analysis
python scripts/bad_case_analysis.py
```
