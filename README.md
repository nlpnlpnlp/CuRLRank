# CuRLRank: Curriculum-Guided Reinforcement Learning for Reasoning-Intensive Reranking

[![Paper Status](https://img.shields.io/badge/Status-ACL--2026--Submission-orange)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.10.13-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)

This repository contains the official implementation of the paper **"CuRLRank: Curriculum-Guided Reinforcement Learning for Reasoning-Intensive Reranking"**. 

> [!NOTE]
> We have released key experimental code. The full codebase will be released upon acceptance.

---

## 1. Environments 🛠️

* **OS:** Ubuntu 22.04.4 LTS
* **GPU:** NVIDIA RTX 6000 Ada
* **CUDA:** 12.4
* **Python:** 3.10.13

We suggest creating a virtual environment with Conda:

```bash
# Create and activate the environment
conda create -n CuRLRank python=3.10.13
conda activate CuRLRank

# Install required packages
pip install -r requirements.txt
```

------

## 2. Datasets 📊

Following previous research, we use the **R2MED** and **BRIGHT** benchmarks.

### 📂 Corpus (Queries and Documents)

| **Dataset** | **Source Link**                                              |
| ----------- | ------------------------------------------------------------ |
| **R2MED**   | [Hugging Face - R2MED](https://huggingface.co/R2MED)         |
| **BRIGHT**  | [Hugging Face - BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) |

### 📂 Retriever Top-100 Results

- **R2MED:** Results are generated using [reason-embed-qwen3-4b-0928](https://huggingface.co/hanhainebula/reason-embed-qwen3-4b-0928).
- **BRIGHT:** [Download from Hugging Face](https://huggingface.co/datasets/hanhainebula/bright-search-results/tree/main/examples/reason-embed-qwen3-8b-0928)

------

## 3. Training 🚀

### 3.1 Supervised Fine-Tuning (SFT)

Setup the environment for [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
cd ./train/LLaMA-Factory/
pip install -e ".[torch,metrics]" --no-build-isolation
```

**Configuration:** Please modify the following parameters in your configuration script:

- `model_name_or_path`: `BASE_MODEL_PATH`
- `dataset`: `DATASET_NAME`
- `output_dir`: `OUTPUT_PATH`

**Run Training:**

```sh
bash run_train.sh
```

### 3.2 Curriculum-Guided Reinforcement Learning

We utilize the [verl](https://github.com/volcengine/verl) framework for GRPO training, loading the initial policy from the SFT stage.

#### 3.2.1 Environment Setup

```
cd ./train/verl/  # Using verl==0.4.0
bash scripts/install_vllm_sglang_mcore.sh
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

#### 3.2.2 GRPO Training

The **Hybrid Reward** is implemented in `verl/verl/utils/reward_score/ranking.py`.

Run the following command to train **CuRLRank (4B)**:

```sh
bash train_grpo.sh
```

------

## 4. Evaluation 📏

To evaluate the CuRLRank (4B) model:

```sh
cd evaluation
bash scripts/curlrank.sh
```

------

## 5. Performance 🏆

Comparison of overall performance (**Average nDCG@10**) on R2MED and BRIGHT benchmarks.

| **Model**           | **Size** | **Strategy** | **R2MED (Avg.)** | **BRIGHT (Avg.)** |
| ------------------- | -------- | ------------ | ---------------- | ----------------- |
| ReasonEmbed         | 4B/8B    | Retriever    | 39.69            | 38.15             |
| RankLLaMA           | 7B       | Pointwise    | 28.96            | 23.05             |
| RankZephyr          | 7B       | Listwise     | 29.36            | 23.27             |
| Rank-R1             | 7B       | Setwise      | 41.11            | 30.55             |
| REARANK             | 7B       | Listwise     | 41.48            | 32.41             |
| ReasonRank          | 7B       | Listwise     | 45.43            | 35.75             |
| Retro*              | 7B       | Pointwise    | 44.31            | 36.39             |
| **CuRLRank (Ours)** | **4B**   | **Listwise** | **48.85**        | **38.32**         |

------



